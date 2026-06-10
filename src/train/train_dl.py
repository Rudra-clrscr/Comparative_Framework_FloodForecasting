"""Shared deep-learning trainer for GRU, LSTM, ME-LSTM."""
import argparse
import csv
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from src.data.dataset import make_loader
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel
from src.models.me_lstm import MELSTMModel
from src.utils.config import (
    DATA_PROCESSED, METRICS_DIR, CONFIGS_DIR, DYNAMIC_FEATURES, STATIC_ATTRIBUTES,
    SEED, DL_DEFAULTS, SEQ_LEN
)
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_dl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nse_loss(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Negative NSE for use as validation metric (higher NSE = lower loss)."""
    return torch.sum((obs - pred) ** 2) / (torch.sum((obs - obs.mean()) ** 2) + 1e-8)


MAX_BASINS_DL = 30  # cap: 208K windows, ~407 batches/epoch


def load_basin_list(processed_dir: str = DATA_PROCESSED, max_basins: int = MAX_BASINS_DL) -> List[str]:
    meta = Path(processed_dir).parent / "metadata" / "basin_list.txt"
    if meta.exists():
        with open(meta) as f:
            basins = [l.strip() for l in f if l.strip()]
    else:
        train_dir = Path(processed_dir) / "train"
        basins = [p.stem for p in train_dir.glob("*.parquet") if not p.stem.endswith("_tabular")]
    if len(basins) > max_basins:
        import random
        random.seed(SEED)
        original_count = len(basins)
        basins = random.sample(basins, max_basins)
        logger.info(f"Sampled {max_basins} basins from {original_count} for DL training")
    return basins


def build_model(model_name: str, n_dynamic: int, n_static: int, cfg: dict) -> nn.Module:
    hs  = cfg.get("hidden_size",     DL_DEFAULTS["hidden_size"])
    nl  = cfg.get("num_layers",      DL_DEFAULTS["num_layers"])
    do  = cfg.get("dropout",         DL_DEFAULTS["dropout"])
    eld = cfg.get("static_embed_dim",DL_DEFAULTS["static_embed_dim"])

    if model_name == "gru":
        return GRUModel(n_dynamic, n_static, hs, nl, do)
    elif model_name == "lstm":
        return LSTMModel(n_dynamic, n_static, hs, nl, do)
    elif model_name == "me_lstm":
        return MELSTMModel(n_dynamic, n_static, eld, hs, nl, do)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_epoch(model, loader, optimizer, scaler, train: bool):
    import time as _time
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    t0 = _time.time()
    with ctx:
        for batch_idx, (x_dyn, y, x_static) in enumerate(loader):
            x_dyn    = x_dyn.to(DEVICE, non_blocking=True)
            y        = y.to(DEVICE, non_blocking=True)
            x_static = x_static.to(DEVICE, non_blocking=True)
            if train:
                optimizer.zero_grad()
            with autocast("cuda", enabled=torch.cuda.is_available()):
                pred = model(x_dyn, x_static)
                loss = criterion(pred, y)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            n += len(y)
            if batch_idx % 200 == 0 and batch_idx > 0:
                elapsed = _time.time() - t0
                logger.info(f"  batch {batch_idx}/{len(loader)}, {elapsed:.0f}s, {elapsed/batch_idx*1000:.1f}ms/b")
    return total_loss / max(n, 1)


def compute_val_nse(model, loader) -> float:
    model.eval()
    all_pred, all_obs = [], []
    with torch.no_grad():
        for x_dyn, y, x_static in loader:
            pred = model(x_dyn.to(DEVICE), x_static.to(DEVICE))
            all_pred.append(pred.cpu().numpy())
            all_obs.append(y.numpy())
    pred_arr = np.concatenate(all_pred)
    obs_arr  = np.concatenate(all_obs)
    denom = np.sum((obs_arr - obs_arr.mean()) ** 2)
    if denom < 1e-8:
        return 0.0
    return float(1 - np.sum((obs_arr - pred_arr) ** 2) / denom)


def main(model_name: str, config_path: str = None):
    set_seed(SEED)
    logger.info(f"Training {model_name.upper()} on device: {DEVICE}")

    cfg = {}
    if config_path is None:
        config_path = Path(CONFIGS_DIR) / f"{model_name}.yaml"
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    basin_list = load_basin_list()
    if not basin_list:
        logger.error("No basins found. Run preprocessing first.")
        return

    static_norm_path = str(Path(DATA_PROCESSED).parent / "metadata" / "static_normalized.parquet")
    has_static = Path(static_norm_path).exists()
    n_static = len(STATIC_ATTRIBUTES) if has_static else 0
    n_dynamic = len(DYNAMIC_FEATURES)

    seq_len    = cfg.get("seq_len",    SEQ_LEN)
    batch_size = cfg.get("batch_size", DL_DEFAULTS["batch_size"])
    lr         = cfg.get("lr",         DL_DEFAULTS["lr"])
    wd         = cfg.get("weight_decay",DL_DEFAULTS["weight_decay"])
    epochs     = cfg.get("epochs",     DL_DEFAULTS["epochs"])
    patience   = cfg.get("patience",   DL_DEFAULTS["patience"])

    train_loader = make_loader("train", basin_list, batch_size=batch_size, shuffle=True,
                               static_norm_path=static_norm_path if has_static else None,
                               seq_len=seq_len)
    val_loader   = make_loader("val",   basin_list, batch_size=batch_size, shuffle=False,
                               static_norm_path=static_norm_path if has_static else None,
                               seq_len=seq_len)

    if len(train_loader) == 0:
        logger.error("Empty train loader. Check data.")
        return

    model = build_model(model_name, n_dynamic, n_static, cfg).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler("cuda", enabled=torch.cuda.is_available())

    out_dir = Path(METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / f"{model_name}_history.csv"

    best_val_nse = -float("inf")
    no_improve   = 0
    t_start      = time.time()

    with open(history_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "train_loss", "val_nse"])

        for epoch in range(1, epochs + 1):
            train_loss = run_epoch(model, train_loader, optimizer, scaler, train=True)
            val_nse    = compute_val_nse(model, val_loader)
            scheduler.step()

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_nse:.4f}"])
            csvf.flush()
            logger.info(f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | val_nse={val_nse:.4f}")

            if val_nse > best_val_nse:
                best_val_nse = val_nse
                no_improve   = 0
                torch.save(model.state_dict(), out_dir / f"{model_name}_best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stop at epoch {epoch} (patience={patience})")
                    break

    convergence_s = time.time() - t_start
    with open(out_dir / f"{model_name}_convergence.txt", "w") as f:
        f.write(str(convergence_s))

    # Save model architecture info for loading at eval time
    arch = dict(
        model_name=model_name,
        n_dynamic=n_dynamic,
        n_static=n_static,
        cfg=cfg,
    )
    import json
    with open(out_dir / f"{model_name}_arch.json", "w") as f:
        json.dump(arch, f)

    logger.info(f"Training complete. Best val NSE={best_val_nse:.4f}, time={convergence_s:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gru", "lstm", "me_lstm"], required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.model, args.config)
