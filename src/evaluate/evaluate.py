"""
Evaluate all five models on the test split.
Produces results/metrics/summary.csv and nse_per_basin.pkl.
"""
import argparse
import gc
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import torch

from src.data.attributes import load_all_attributes, get_static_features, normalize_static
from src.evaluate.metrics import nse, kge, mpte, pbias_hf, measure_latency, model_memory_mb, sklearn_memory_mb
from src.models.gru import GRUModel
from src.models.lstm import LSTMModel
from src.models.me_lstm import MELSTMModel
from src.utils.config import (
    DATA_PROCESSED, DATA_RAW, METRICS_DIR, DYNAMIC_FEATURES, TARGET_COL, SEQ_LEN, SEED
)
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger("evaluate")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_basin_list(processed_dir: str) -> List[str]:
    meta = Path(processed_dir).parent / "metadata" / "basin_list.txt"
    if meta.exists():
        with open(meta) as f:
            return [l.strip() for l in f if l.strip()]
    return [p.stem for p in (Path(processed_dir) / "test").glob("*.parquet")
            if not p.stem.endswith("_tabular")]


def load_basin_stats(processed_dir: str) -> Dict:
    p = Path(processed_dir).parent / "metadata" / "basin_stats.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def denorm_target(arr: np.ndarray, stats: Dict) -> np.ndarray:
    return arr * stats["target"]["std"] + stats["target"]["mean"]


def load_static_norm(processed_dir: str):
    p = Path(processed_dir).parent / "metadata" / "static_normalized.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def load_dl_model(model_name: str, metrics_dir: str) -> torch.nn.Module:
    arch_path = Path(metrics_dir) / f"{model_name}_arch.json"
    pt_path   = Path(metrics_dir) / f"{model_name}_best.pt"
    if not arch_path.exists() or not pt_path.exists():
        return None
    with open(arch_path) as f:
        arch = json.load(f)
    model_name_key = arch["model_name"]
    n_dyn   = arch["n_dynamic"]
    n_stat  = arch["n_static"]
    cfg     = arch.get("cfg", {})
    hs  = cfg.get("hidden_size", 256)
    nl  = cfg.get("num_layers",  2)
    do  = cfg.get("dropout",     0.4)
    eld = cfg.get("static_embed_dim", 32)
    if model_name_key == "gru":
        m = GRUModel(n_dyn, n_stat, hs, nl, do)
    elif model_name_key == "lstm":
        m = LSTMModel(n_dyn, n_stat, hs, nl, do)
    else:
        m = MELSTMModel(n_dyn, n_stat, eld, hs, nl, do)
    m.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m


def predict_dl_basin(
    model, basin_id: str, processed_dir: str, static_norm_df: pd.DataFrame,
    seq_len: int = SEQ_LEN, batch_size: int = 512,
) -> np.ndarray:
    pq = Path(processed_dir) / "test" / f"{basin_id}.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    dyn_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
    x_dyn = df[dyn_cols].values.astype(np.float32)
    if len(x_dyn) <= seq_len:
        return None

    static_np = np.zeros(1, dtype=np.float32)
    if not static_norm_df.empty and basin_id in static_norm_df.index:
        static_np = static_norm_df.loc[basin_id].values.astype(np.float32)

    n_preds = len(x_dyn) - seq_len
    # Vectorized window creation: (n_preds, n_features, seq_len) → (n_preds, seq_len, n_features)
    # sliding_window_view is a zero-copy C-level op; ascontiguousarray makes one copy for GPU transfer
    all_windows = np.ascontiguousarray(
        sliding_window_view(x_dyn, seq_len, axis=0).transpose(0, 2, 1)
    )
    x_stat_t = torch.from_numpy(static_np).unsqueeze(0).to(DEVICE)
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_preds, batch_size):
            end = min(start + batch_size, n_preds)
            x_seq = torch.from_numpy(all_windows[start:end]).to(DEVICE)
            out = model(x_seq, x_stat_t.expand(end - start, -1))
            preds.append(out.cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def predict_ml_basin(
    model, basin_id: str, processed_dir: str, feature_cols: List[str]
) -> np.ndarray:
    pq = Path(processed_dir) / "test" / f"{basin_id}_tabular.parquet"
    if not pq.exists():
        return None, None
    df = pd.read_parquet(pq).dropna()
    feat_cols = [c for c in feature_cols if c in df.columns]
    X = df[feat_cols].values
    y = df[TARGET_COL].values
    preds = model.predict(X)
    return preds, y, df


def load_convergence(model_name: str, metrics_dir: str) -> float:
    p = Path(metrics_dir) / f"{model_name}_convergence.txt"
    if p.exists():
        return float(p.read_text().strip())
    return np.nan


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_all(results_dir: str = METRICS_DIR, processed_dir: str = DATA_PROCESSED):
    set_seed(SEED)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    basin_list   = load_basin_list(processed_dir)
    basin_stats  = load_basin_stats(processed_dir)
    static_norm  = load_static_norm(processed_dir)

    logger.info(f"Evaluating on {len(basin_list)} basins")

    rows = []
    nse_per_basin = {}

    # -------- RF --------
    rf_path = results_dir / "rf_model.pkl"
    if rf_path.exists():
        logger.info("Evaluating RF...")
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)

        feat_cols = None
        nses, kges, mptes, pbias_hfs = [], [], [], []
        all_X_test = []
        for bid in basin_list:
            pq = Path(processed_dir) / "test" / f"{bid}_tabular.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq).dropna()
            if feat_cols is None:
                feat_cols = [c for c in df.columns if c != TARGET_COL]
            X = df[[c for c in feat_cols if c in df.columns]].values
            y_norm = df[TARGET_COL].values
            stats = basin_stats.get(bid, {})
            if not stats:
                continue
            pred_norm = rf_model.predict(X)
            obs = denorm_target(y_norm,  stats)
            sim = denorm_target(pred_norm, stats)
            all_X_test.append(X[:100])
            nses.append(nse(obs, sim));  kges.append(kge(obs, sim))
            mptes.append(mpte(obs, sim)); pbias_hfs.append(pbias_hf(obs, sim))

        nse_per_basin["RF"] = np.array([v for v in nses if not np.isnan(v)])
        X_lat = np.concatenate(all_X_test) if all_X_test else np.zeros((1, 1))
        lat_med, _ = measure_latency(rf_model.predict, X_lat[:100], n_repeats=20)
        mem = sklearn_memory_mb(rf_model)
        conv = load_convergence("rf", str(results_dir))
        rows.append(dict(
            model="RF",
            NSE_median=np.nanmedian(nses), NSE_IQR=float(np.subtract(*np.nanpercentile(nses, [75,25]))),
            KGE_median=np.nanmedian(kges), KGE_IQR=float(np.subtract(*np.nanpercentile(kges, [75,25]))),
            MPTE_median=np.nanmedian(mptes), PBIAS_HF_median=np.nanmedian(pbias_hfs),
            latency_ms=lat_med, memory_MB=mem, convergence_s=conv,
        ))
        del rf_model, all_X_test, X_lat
        gc.collect()

    # -------- XGBoost --------
    xgb_path = results_dir / "xgb_model.pkl"
    if xgb_path.exists():
        logger.info("Evaluating XGBoost...")
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)

        feat_cols = None
        nses, kges, mptes, pbias_hfs = [], [], [], []
        all_X_test = []
        for bid in basin_list:
            pq = Path(processed_dir) / "test" / f"{bid}_tabular.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq).dropna()
            if feat_cols is None:
                feat_cols = [c for c in df.columns if c != TARGET_COL]
            X = df[[c for c in feat_cols if c in df.columns]].values
            y_norm = df[TARGET_COL].values
            stats = basin_stats.get(bid, {})
            if not stats:
                continue
            pred_norm = xgb_model.predict(X)
            obs = denorm_target(y_norm,   stats)
            sim = denorm_target(pred_norm, stats)
            all_X_test.append(X[:100])
            nses.append(nse(obs, sim));  kges.append(kge(obs, sim))
            mptes.append(mpte(obs, sim)); pbias_hfs.append(pbias_hf(obs, sim))

        nse_per_basin["XGBoost"] = np.array([v for v in nses if not np.isnan(v)])
        X_lat = np.concatenate(all_X_test) if all_X_test else np.zeros((1, 1))
        lat_med, _ = measure_latency(xgb_model.predict, X_lat[:100], n_repeats=20)
        mem = sklearn_memory_mb(xgb_model)
        conv = load_convergence("xgb", str(results_dir))
        rows.append(dict(
            model="XGBoost",
            NSE_median=np.nanmedian(nses), NSE_IQR=float(np.subtract(*np.nanpercentile(nses, [75,25]))),
            KGE_median=np.nanmedian(kges), KGE_IQR=float(np.subtract(*np.nanpercentile(kges, [75,25]))),
            MPTE_median=np.nanmedian(mptes), PBIAS_HF_median=np.nanmedian(pbias_hfs),
            latency_ms=lat_med, memory_MB=mem, convergence_s=conv,
        ))
        del xgb_model, all_X_test, X_lat
        gc.collect()

    # -------- DL models --------
    for model_key, display_name in [("gru","GRU"),("lstm","LSTM"),("me_lstm","ME-LSTM")]:
        dl_model = load_dl_model(model_key, str(results_dir))
        if dl_model is None:
            logger.warning(f"{display_name} model not found, skipping.")
            continue
        logger.info(f"Evaluating {display_name}...")

        nses, kges, mptes, pbias_hfs = [], [], [], []
        for b_idx, bid in enumerate(basin_list):
            if b_idx % 500 == 0:
                logger.info(f"  {display_name}: basin {b_idx}/{len(basin_list)}")
            pred_norm = predict_dl_basin(dl_model, bid, processed_dir, static_norm)
            if pred_norm is None:
                continue
            pq = Path(processed_dir) / "test" / f"{bid}.parquet"
            df = pd.read_parquet(pq)
            y_norm = df[TARGET_COL].values[SEQ_LEN:]
            if len(y_norm) != len(pred_norm):
                min_len = min(len(y_norm), len(pred_norm))
                y_norm   = y_norm[:min_len]
                pred_norm = pred_norm[:min_len]
            stats = basin_stats.get(bid, {})
            if not stats:
                continue
            obs = denorm_target(y_norm,   stats)
            sim = denorm_target(pred_norm, stats)
            nses.append(nse(obs, sim));  kges.append(kge(obs, sim))
            mptes.append(mpte(obs, sim)); pbias_hfs.append(pbias_hf(obs, sim))

        nse_per_basin[display_name] = np.array([v for v in nses if not np.isnan(v)])

        # Latency on one basin
        sample_bid = basin_list[0]
        pq = Path(processed_dir) / "test" / f"{sample_bid}.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
            dyn_cols = [c for c in DYNAMIC_FEATURES if c in df.columns]
            x_dyn = torch.from_numpy(df[dyn_cols].values[:SEQ_LEN].astype(np.float32)).unsqueeze(0).to(DEVICE)
            static_np = np.zeros(1, dtype=np.float32)
            if not static_norm.empty and sample_bid in static_norm.index:
                static_np = static_norm.loc[sample_bid].values.astype(np.float32)
            x_stat = torch.from_numpy(static_np).unsqueeze(0).to(DEVICE)
            def predict_fn(_):
                with torch.no_grad():
                    dl_model(x_dyn, x_stat)
            lat_med, _ = measure_latency(predict_fn, None, n_repeats=50)
        else:
            lat_med = np.nan

        mem = model_memory_mb(dl_model)
        conv = load_convergence(model_key, str(results_dir))
        rows.append(dict(
            model=display_name,
            NSE_median=np.nanmedian(nses), NSE_IQR=float(np.subtract(*np.nanpercentile(nses, [75,25]))),
            KGE_median=np.nanmedian(kges), KGE_IQR=float(np.subtract(*np.nanpercentile(kges, [75,25]))),
            MPTE_median=np.nanmedian(mptes), PBIAS_HF_median=np.nanmedian(pbias_hfs),
            latency_ms=lat_med, memory_MB=mem, convergence_s=conv,
        ))
        del dl_model
        gc.collect()
        torch.cuda.empty_cache()

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(results_dir / "summary.csv", index=False)
        logger.info(f"\n{summary.to_string(index=False)}")
        logger.info(f"Summary saved to {results_dir / 'summary.csv'}")

    with open(results_dir / "nse_per_basin.pkl", "wb") as f:
        pickle.dump(nse_per_basin, f)
    logger.info("nse_per_basin.pkl saved.")

    # Save raw predictions for hydrograph plotting (best/median/worst NSE basins)
    _save_hydrograph_data(basin_list, basin_stats, processed_dir, results_dir, static_norm, nse_per_basin)


def _save_hydrograph_data(basin_list, basin_stats, processed_dir, results_dir, static_norm, nse_per_basin):
    """Save obs + model predictions for 3 representative basins (first/mid/last valid)."""
    valid_basins = [
        b for b in basin_list
        if (Path(processed_dir) / "test" / f"{b}.parquet").exists() and basin_stats.get(b)
    ]
    if len(valid_basins) < 3:
        return

    n = len(valid_basins)
    selected = [valid_basins[0], valid_basins[n // 2], valid_basins[-1]]

    # Reload DL models for inference on the 3 selected basins
    dl_models = {}
    for model_key, display in [("gru", "GRU"), ("lstm", "LSTM"), ("me_lstm", "ME-LSTM")]:
        m = load_dl_model(model_key, str(results_dir))
        if m is not None:
            dl_models[display] = m

    # Reload ML models; build feature column list
    ml_models = {}
    feat_cols = None
    rf_path = results_dir / "rf_model.pkl"
    xgb_path = results_dir / "xgb_model.pkl"
    if rf_path.exists():
        with open(rf_path, "rb") as f:
            ml_models["RF"] = pickle.load(f)
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            ml_models["XGBoost"] = pickle.load(f)
    for bid in valid_basins:
        tp = Path(processed_dir) / "test" / f"{bid}_tabular.parquet"
        if tp.exists():
            feat_cols = [c for c in pd.read_parquet(tp).columns if c != TARGET_COL]
            break

    hydro_data = {}
    for bid in selected:
        pq = Path(processed_dir) / "test" / f"{bid}.parquet"
        df_dyn = pd.read_parquet(pq)
        stats = basin_stats[bid]

        y_norm = df_dyn[TARGET_COL].values[SEQ_LEN:]
        obs = denorm_target(y_norm, stats)
        dates = (pd.to_datetime(df_dyn["date"].values[SEQ_LEN:])
                 if "date" in df_dyn.columns else list(range(len(obs))))

        preds = {}
        for name, model in dl_models.items():
            pred_norm = predict_dl_basin(model, bid, processed_dir, static_norm)
            if pred_norm is not None:
                n_match = min(len(obs), len(pred_norm))
                preds[name] = denorm_target(pred_norm[:n_match], stats)

        # ML: align to the DL SEQ_LEN offset by taking rows [SEQ_LEN:] from full tabular
        if feat_cols is not None:
            tp = Path(processed_dir) / "test" / f"{bid}_tabular.parquet"
            if tp.exists():
                df_tab = pd.read_parquet(tp)
                X_all = df_tab[[c for c in feat_cols if c in df_tab.columns]].values
                valid_mask = ~np.any(np.isnan(X_all), axis=1)
                pred_full = np.full(len(X_all), np.nan, dtype=np.float32)
                for name, ml_model in ml_models.items():
                    pred_full[valid_mask] = ml_model.predict(X_all[valid_mask])
                    ml_slice = pred_full[SEQ_LEN: SEQ_LEN + len(obs)]
                    if len(ml_slice) > 0:
                        preds[name] = denorm_target(ml_slice, stats)

        hydro_data[bid] = {"obs": obs, "preds": preds, "dates": dates}

    with open(results_dir / "hydrograph_data.pkl", "wb") as f:
        pickle.dump(hydro_data, f)
    with open(results_dir / "hydrograph_basins.txt", "w") as f:
        f.write("\n".join(selected))
    logger.info(f"Hydrograph data saved for {len(hydro_data)} basins.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",   default=METRICS_DIR)
    parser.add_argument("--processed_dir", default=DATA_PROCESSED)
    args = parser.parse_args()
    evaluate_all(args.results_dir, args.processed_dir)
