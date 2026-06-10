"""Hydrological and compute metrics."""
import statistics
import time
import tracemalloc
from typing import Callable

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Hydrological skill scores
# ---------------------------------------------------------------------------

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom < 1e-8:
        return np.nan
    return float(1 - np.sum((obs - sim) ** 2) / denom)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if np.std(obs) < 1e-8 or np.std(sim) < 1e-8:
        return np.nan
    r     = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(np.std(sim) / np.std(obs))
    beta  = float(np.mean(sim) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    if np.isnan(beta):
        return np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def mpte(obs: np.ndarray, sim: np.ndarray, window: int = 5) -> float:
    """Mean Peak Timing Error (days)."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if len(obs) < 2 * window + 1:
        return np.nan

    prom = max(float(np.std(obs)) * 0.5, 1e-6)
    obs_peaks, _ = find_peaks(obs, prominence=prom, distance=window)
    sim_peaks, _ = find_peaks(sim, prominence=prom, distance=window)

    if len(obs_peaks) == 0 or len(sim_peaks) == 0:
        return np.nan

    errors = []
    for op in obs_peaks:
        diffs = np.abs(sim_peaks - op)
        nearest = int(diffs.min())
        errors.append(nearest)

    return float(np.mean(errors))


def pbias_hf(obs: np.ndarray, sim: np.ndarray, quantile: float = 0.98) -> float:
    """Percent bias of top 2% flows."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    threshold = np.quantile(obs, quantile)
    idx = obs >= threshold
    if idx.sum() == 0:
        return np.nan
    obs_hf = obs[idx]
    sim_hf = sim[idx]
    denom = np.sum(obs_hf)
    if denom < 1e-8:
        return np.nan
    return float(100 * np.sum(sim_hf - obs_hf) / denom)


# ---------------------------------------------------------------------------
# Compute efficiency
# ---------------------------------------------------------------------------

def measure_latency(predict_fn: Callable, X, n_repeats: int = 100) -> tuple:
    """Returns (median_ms, std_ms)."""
    latencies = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        latencies.append((time.perf_counter() - t0) * 1000)
    return statistics.median(latencies), statistics.stdev(latencies)


def model_memory_mb(model) -> float:
    """PyTorch model parameter memory in MB."""
    try:
        import torch
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    except Exception:
        return _pickle_size_mb(model)


def _pickle_size_mb(obj) -> float:
    import pickle, io
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.tell() / 1e6


def sklearn_memory_mb(model) -> float:
    return _pickle_size_mb(model)
