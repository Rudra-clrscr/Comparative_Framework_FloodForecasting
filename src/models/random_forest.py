from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
from pathlib import Path
from src.utils.logging_utils import get_logger

logger = get_logger("rf_model")


def build_rf(
    n_estimators: int = 500,
    max_depth=None,
    min_samples_leaf: int = 5,
    max_features: float = 0.5,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def save_rf(model: RandomForestRegressor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"RF model saved to {path}")


def load_rf(path: str) -> RandomForestRegressor:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
