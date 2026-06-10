"""Train XGBoost on all basins' tabular features with val early stopping."""
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.xgboost_model import build_xgb, save_xgb
from src.utils.config import DATA_PROCESSED, METRICS_DIR, CONFIGS_DIR, TARGET_COL, SEED
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_xgb")


MAX_BASINS_ML = 200  # cap for memory (~1.6 GB RAM)


def load_tabular(split: str, processed_dir: str = DATA_PROCESSED, max_basins: int = MAX_BASINS_ML):
    split_dir = Path(processed_dir) / split
    all_files = sorted(split_dir.glob("*_tabular.parquet"))
    if len(all_files) > max_basins:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(all_files), max_basins, replace=False)
        all_files = [all_files[i] for i in sorted(idx)]
    logger.info(f"Loading {len(all_files)} basin tabular files for [{split}] split...")
    frames = []
    for p in all_files:
        df = pd.read_parquet(p)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), np.array([])
    data = pd.concat(frames, ignore_index=True).dropna()
    target = data.pop(TARGET_COL).values
    return data, target


def main(config_path: str = None):
    set_seed(SEED)

    cfg = {}
    if config_path is None:
        config_path = Path(CONFIGS_DIR) / "xgb.yaml"
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    logger.info("Loading tabular data...")
    X_train, y_train = load_tabular("train")
    X_val,   y_val   = load_tabular("val")

    if len(X_train) == 0:
        logger.error("No training data. Run preprocessing first.")
        return

    # Align columns
    common_cols = X_train.columns.intersection(X_val.columns)
    X_train = X_train[common_cols]
    X_val   = X_val[common_cols]

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    model = build_xgb(
        n_estimators=cfg.get("n_estimators", 1000),
        learning_rate=cfg.get("learning_rate", 0.01),
        max_depth=cfg.get("max_depth", 6),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
        random_state=cfg.get("random_state", SEED),
    )

    t0 = time.time()
    logger.info("Fitting XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    convergence_s = time.time() - t0
    logger.info(f"XGBoost training complete in {convergence_s:.1f}s, best iter: {model.best_iteration}")

    out_dir = Path(METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_xgb(model, str(out_dir / "xgb_model.pkl"))

    with open(out_dir / "xgb_convergence.txt", "w") as f:
        f.write(str(convergence_s))

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
