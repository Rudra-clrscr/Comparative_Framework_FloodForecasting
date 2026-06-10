"""PyTorch Dataset and DataLoader wrappers for DL models."""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import (
    DATA_PROCESSED, SEQ_LEN, DYNAMIC_FEATURES, TARGET_COL, SEED
)
from src.utils.logging_utils import get_logger

logger = get_logger("dataset")


class BasinDataset(Dataset):
    """Sliding-window dataset for a single basin."""

    def __init__(
        self,
        parquet_path: str,
        static_features: Optional[np.ndarray],
        seq_len: int = SEQ_LEN,
    ):
        df = pd.read_parquet(parquet_path)
        dyn_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
        self.x_dyn    = df[dyn_cols].values.astype(np.float32)
        self.y        = df[TARGET_COL].values.astype(np.float32)
        self.static   = static_features if static_features is not None else np.zeros(1, dtype=np.float32)
        self.seq_len  = seq_len
        self.n_dyn    = len(dyn_cols)

    def __len__(self) -> int:
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_seq = self.x_dyn[idx: idx + self.seq_len]          # (seq_len, n_dyn)
        y_val = self.y[idx + self.seq_len]                     # scalar
        return (
            torch.from_numpy(np.ascontiguousarray(x_seq)),
            torch.tensor(y_val, dtype=torch.float32),
            torch.from_numpy(self.static),
        )


class MultiBasinDataset(Dataset):
    """Concatenated dataset across all basins using cumulative-index lookup
    instead of a flat Python list of (dataset, idx) tuples — avoids 3M+ object overhead."""

    def __init__(
        self,
        split: str,
        basin_list: List[str],
        static_norm_path: Optional[str] = None,
        seq_len: int = SEQ_LEN,
        processed_dir: str = DATA_PROCESSED,
    ):
        self.seq_len  = seq_len
        self._datasets: List[BasinDataset] = []
        cumlen = [0]

        static_lookup = {}
        if static_norm_path and Path(static_norm_path).exists():
            static_df = pd.read_parquet(static_norm_path)
            static_lookup = {bid: static_df.loc[bid].values.astype(np.float32)
                             for bid in static_df.index if bid in set(basin_list)}

        split_dir = Path(processed_dir) / split
        for basin_id in basin_list:
            pq = split_dir / f"{basin_id}.parquet"
            if not pq.exists():
                continue
            static = static_lookup.get(basin_id, np.zeros(1, dtype=np.float32))
            ds = BasinDataset(str(pq), static, seq_len=seq_len)
            n = len(ds)
            if n == 0:
                continue
            self._datasets.append(ds)
            cumlen.append(cumlen[-1] + n)

        self._cumlen = np.array(cumlen, dtype=np.int64)
        logger.info(f"MultiBasinDataset [{split}]: {self._cumlen[-1]} windows from {len(self._datasets)} basins")

    def __len__(self) -> int:
        return int(self._cumlen[-1])

    def __getitem__(self, idx: int):
        # Binary search: O(log N_basins) instead of Python list of 3M tuples
        basin_idx = int(np.searchsorted(self._cumlen[1:], idx, side='right'))
        local_idx = idx - int(self._cumlen[basin_idx])
        return self._datasets[basin_idx][local_idx]


def make_loader(
    split: str,
    basin_list: List[str],
    batch_size: int = 256,
    shuffle: bool = True,
    static_norm_path: Optional[str] = None,
    seq_len: int = SEQ_LEN,
    processed_dir: str = DATA_PROCESSED,
    num_workers: int = 0,
) -> DataLoader:
    ds = MultiBasinDataset(split, basin_list, static_norm_path, seq_len, processed_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
