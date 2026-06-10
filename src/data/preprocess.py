"""
Preprocessing pipeline: load basins, clean, normalize, split, build features.
Outputs processed splits to data/processed/{train,val,test}/.
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.data.attributes import load_all_attributes, get_static_features, normalize_static
from src.utils.config import (
    DATA_RAW, DATA_PROCESSED, SEED,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    DYNAMIC_FEATURES, TARGET_COL, MAX_NAN_FRACTION, MAX_GAP_FILL_DAYS, LAG_DAYS,
)
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("preprocess")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_caravan_root(raw_dir: str) -> Path:
    """Find the Caravan root (where timeseries/ and attributes/ live)."""
    root = Path(raw_dir)
    # Direct match
    if (root / "timeseries" / "csv").exists():
        return root
    # Deep search (archive may have an absolute path prefix)
    for candidate in sorted(root.rglob("timeseries")):
        ts = candidate / "csv"
        if ts.exists():
            caravan_root = candidate.parent
            logger.info(f"Auto-detected Caravan root: {caravan_root}")
            return caravan_root
    return root


def discover_basin_files(raw_dir: str = DATA_RAW) -> Dict[str, Path]:
    caravan_root = _find_caravan_root(raw_dir)
    ts_dir = caravan_root / "timeseries" / "csv"
    basins = {}
    if not ts_dir.exists():
        logger.warning(f"Timeseries directory not found: {ts_dir}")
        return basins
    for csv_path in sorted(ts_dir.rglob("*.csv")):
        basin_id = csv_path.stem
        basins[basin_id] = csv_path
    logger.info(f"Discovered {len(basins)} basin files under {ts_dir}")
    return basins


def load_basin(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def gap_fill(series: pd.Series, max_gap: int = MAX_GAP_FILL_DAYS) -> pd.Series:
    """Linear interpolation for gaps ≤ max_gap days."""
    filled = series.copy()
    null_mask = filled.isnull()
    if not null_mask.any():
        return filled
    # label consecutive runs of NaN
    groups = (null_mask != null_mask.shift()).cumsum()[null_mask]
    for gid, grp in groups.groupby(groups):
        if len(grp) <= max_gap:
            filled.loc[grp.index] = np.nan  # keep as-is for interpolate
    filled = filled.interpolate(method="time", limit=max_gap)
    return filled


def clean_basin(df: pd.DataFrame, basin_id: str) -> Optional[pd.DataFrame]:
    if TARGET_COL not in df.columns:
        logger.warning(f"{basin_id}: no '{TARGET_COL}' column, skipping")
        return None

    # Check NaN fraction only within the study period (not the full ERA5 record)
    study = df.loc[TRAIN_START:TEST_END]
    if len(study) == 0:
        return None
    target_nan_frac = study[TARGET_COL].isnull().mean()
    if target_nan_frac > MAX_NAN_FRACTION:
        logger.debug(f"{basin_id}: streamflow NaN fraction {target_nan_frac:.2%} in study period, dropping")
        return None

    df = df.copy()
    df[TARGET_COL] = gap_fill(df[TARGET_COL])
    for col in DYNAMIC_FEATURES:
        if col in df.columns:
            df[col] = gap_fill(df[col])

    df = df.dropna(subset=[TARGET_COL])
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_basin(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df.loc[TRAIN_START:TRAIN_END].copy()
    val   = df.loc[VAL_START:VAL_END].copy()
    test  = df.loc[TEST_START:TEST_END].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_basin_stats(train_df: pd.DataFrame) -> Dict:
    cols = [c for c in DYNAMIC_FEATURES if c in train_df.columns]
    stats = {
        "dynamic": {
            "mean": train_df[cols].mean(),
            "std":  train_df[cols].std().replace(0, 1),
        },
        "target": {
            "mean": float(train_df[TARGET_COL].mean()),
            "std":  float(train_df[TARGET_COL].std()) or 1.0,
        },
    }
    return stats


def normalize_split(df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
    df[cols] = (df[cols] - stats["dynamic"]["mean"]) / stats["dynamic"]["std"]
    df[TARGET_COL] = (df[TARGET_COL] - stats["target"]["mean"]) / stats["target"]["std"]
    return df


# ---------------------------------------------------------------------------
# Tabular feature construction (RF / XGBoost)
# ---------------------------------------------------------------------------

def build_tabular_features(df: pd.DataFrame, static_row: Optional[pd.Series] = None) -> pd.DataFrame:
    dyn_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
    rows = []
    for lag in LAG_DAYS:
        for col in dyn_cols + [TARGET_COL]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for window in [7, 14, 30]:
        for col in dyn_cols:
            df[f"{col}_roll{window}_mean"] = df[col].rolling(window).mean()
            df[f"{col}_roll{window}_std"]  = df[col].rolling(window).std()

    df = df.dropna()

    if static_row is not None:
        for k, v in static_row.items():
            df[f"static_{k}"] = v

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(raw_dir: str = DATA_RAW, out_dir: str = DATA_PROCESSED):
    set_seed(SEED)
    out_dir = Path(out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    # Load static attributes
    attr_df = load_all_attributes(raw_dir)
    static_df = get_static_features(attr_df) if not attr_df.empty else pd.DataFrame()
    static_norm, static_stats = normalize_static(static_df) if not static_df.empty else (pd.DataFrame(), {})

    basin_files = discover_basin_files(raw_dir)
    if not basin_files:
        logger.error("No basin files found. Did you run extract.py first?")
        return

    all_basin_stats = {}
    kept_basins = []
    dropped_basins = []
    total = len(basin_files)

    for i, (basin_id, path) in enumerate(basin_files.items()):
        if (i + 1) % 500 == 0 or i == 0:
            logger.info(f"  Processing basin {i+1}/{total} (kept={len(kept_basins)}, dropped={len(dropped_basins)})")
        df_raw = load_basin(path)
        if df_raw is None:
            dropped_basins.append(basin_id)
            continue

        df_clean = clean_basin(df_raw, basin_id)
        if df_clean is None:
            dropped_basins.append(basin_id)
            continue

        train_df, val_df, test_df = split_basin(df_clean)

        if len(train_df) < 365 or len(test_df) < 30:
            logger.debug(f"{basin_id}: insufficient data after split, dropping")
            dropped_basins.append(basin_id)
            continue

        stats = compute_basin_stats(train_df)
        all_basin_stats[basin_id] = stats

        train_norm = normalize_split(train_df, stats)
        val_norm   = normalize_split(val_df,   stats)
        test_norm  = normalize_split(test_df,  stats)

        static_row = static_norm.loc[basin_id] if (not static_norm.empty and basin_id in static_norm.index) else None

        # Save normalized sequences (DL)
        train_norm.to_parquet(out_dir / "train" / f"{basin_id}.parquet")
        val_norm.to_parquet(out_dir / "val"   / f"{basin_id}.parquet")
        test_norm.to_parquet(out_dir / "test"  / f"{basin_id}.parquet")

        # Save tabular features (ML)
        train_tab = build_tabular_features(train_norm.copy(), static_row)
        val_tab   = build_tabular_features(val_norm.copy(),   static_row)
        test_tab  = build_tabular_features(test_norm.copy(),  static_row)

        train_tab.to_parquet(out_dir / "train" / f"{basin_id}_tabular.parquet")
        val_tab.to_parquet(out_dir / "val"   / f"{basin_id}_tabular.parquet")
        test_tab.to_parquet(out_dir / "test"  / f"{basin_id}_tabular.parquet")

        kept_basins.append(basin_id)

    # Save metadata
    meta_dir = Path(raw_dir).parent / "metadata"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "basin_stats.pkl", "wb") as f:
        pickle.dump(all_basin_stats, f)
    if not static_norm.empty:
        with open(meta_dir / "static_stats.pkl", "wb") as f:
            pickle.dump(static_stats, f)
        static_norm.to_parquet(meta_dir / "static_normalized.parquet")

    with open(meta_dir / "basin_list.txt", "w") as f:
        f.write("\n".join(kept_basins))

    logger.info(f"Preprocessing complete. Kept: {len(kept_basins)}, Dropped: {len(dropped_basins)}")
    logger.info(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",  default=DATA_RAW)
    parser.add_argument("--output",   default=DATA_PROCESSED)
    args = parser.parse_args()
    run_preprocessing(args.raw_dir, args.output)
