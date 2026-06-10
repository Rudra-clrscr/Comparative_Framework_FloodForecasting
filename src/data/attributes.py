import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.config import DATA_RAW, STATIC_ATTRIBUTES
from src.utils.logging_utils import get_logger

logger = get_logger("attributes")

ATTRIBUTE_COLS = STATIC_ATTRIBUTES


def _find_attr_dir(raw_dir: str) -> Path:
    root = Path(raw_dir)
    if (root / "attributes").exists():
        return root / "attributes"
    for candidate in sorted(root.rglob("attributes")):
        if candidate.is_dir() and any(candidate.glob("**/*.csv")):
            return candidate
    return root / "attributes"


def _load_attr_group(attr_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(attr_dir.rglob(pattern)):
        try:
            df = pd.read_csv(csv_path)
            if "gauge_id" in df.columns:
                df = df.set_index("gauge_id")
            frames.append(df)
            logger.info(f"Loaded {csv_path.name}: {len(df)} basins, {len(df.columns)} cols")
        except Exception as e:
            logger.warning(f"Failed to load {csv_path}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0)


def load_all_attributes(raw_dir: str = DATA_RAW) -> pd.DataFrame:
    attr_dir = _find_attr_dir(raw_dir)
    if not attr_dir.exists():
        logger.warning(f"Attributes directory not found: {attr_dir}")
        return pd.DataFrame()

    caravan   = _load_attr_group(attr_dir, "attributes_caravan_*.csv")
    other     = _load_attr_group(attr_dir, "attributes_other_*.csv")
    hydroatlas = _load_attr_group(attr_dir, "attributes_hydroatlas_*.csv")

    parts = [p for p in [caravan, other, hydroatlas] if not p.empty]
    if not parts:
        return pd.DataFrame()

    combined = parts[0]
    for part in parts[1:]:
        combined = combined.join(part, how="outer", rsuffix="_dup")
        dup_cols = [c for c in combined.columns if c.endswith("_dup")]
        combined.drop(columns=dup_cols, inplace=True)

    combined.index.name = "gauge_id"
    logger.info(f"Combined attribute table: {combined.shape}")
    return combined


def get_static_features(attr_df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ATTRIBUTE_COLS if c in attr_df.columns]
    missing = [c for c in ATTRIBUTE_COLS if c not in attr_df.columns]
    if missing:
        logger.warning(f"Missing static attribute columns: {missing}")
    return attr_df[available].copy()


def normalize_static(
    attr_df: pd.DataFrame,
    stats: Optional[Dict] = None
) -> tuple:
    """Global z-score normalization across all basins. Returns (normalized_df, stats_dict)."""
    if stats is None:
        stats = {
            "mean": attr_df.mean(),
            "std":  attr_df.std().replace(0, 1),
        }
    normalized = (attr_df - stats["mean"]) / stats["std"]
    normalized = normalized.fillna(0.0)
    return normalized, stats
