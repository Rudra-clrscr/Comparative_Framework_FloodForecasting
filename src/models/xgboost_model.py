import pickle
from pathlib import Path

import torch
from xgboost import XGBRegressor

from src.utils.logging_utils import get_logger

logger = get_logger("xgb_model")


def build_xgb(
    n_estimators: int = 1000,
    learning_rate: float = 0.01,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> XGBRegressor:
    tree_method = "hist"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
        tree_method=tree_method,
        device=device,
        random_state=random_state,
        verbosity=0,
    )


def save_xgb(model: XGBRegressor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"XGB model saved to {path}")


def load_xgb(path: str) -> XGBRegressor:
    with open(path, "rb") as f:
        return pickle.load(f)
