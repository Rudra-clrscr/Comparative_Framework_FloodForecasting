"""Train Random Forest on all basins' tabular features."""
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.random_forest import build_rf, save_rf
from src.utils.config import DATA_PROCESSED, METRICS_DIR, CONFIGS_DIR, TARGET_COL, SEED
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_rf")


MAX_BASINS_ML = 50  # cap for speed (~363K rows, fits quickly)


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
        config_path = Path(CONFIGS_DIR) / "rf.yaml"
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    logger.info("Loading tabular train data...")
    X_train, y_train = load_tabular("train")
    if len(X_train) == 0:
        logger.error("No training data found. Run preprocessing first.")
        return

    logger.info(f"Train samples: {len(X_train)}, features: {X_train.shape[1]}")

    model = build_rf(
        n_estimators=cfg.get("n_estimators", 500),
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=cfg.get("min_samples_leaf", 5),
        max_features=cfg.get("max_features", 0.5),
        random_state=cfg.get("random_state", SEED),
    )

    t0 = time.time()
    logger.info("Fitting Random Forest...")
    model.fit(X_train, y_train)
    convergence_s = time.time() - t0
    logger.info(f"RF training complete in {convergence_s:.1f}s")

    out_dir = Path(METRICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rf(model, str(out_dir / "rf_model.pkl"))

    with open(out_dir / "rf_convergence.txt", "w") as f:
        f.write(str(convergence_s))

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
