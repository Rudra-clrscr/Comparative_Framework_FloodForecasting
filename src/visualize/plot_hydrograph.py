import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

STYLES = {
    'RF':      ('#378ADD', '--'),
    'XGBoost': ('#BA7517', ':'),
    'GRU':     ('#1D9E75', '-.'),
    'LSTM':    ('#7F77DD', '--'),
    'ME-LSTM': ('#D85A30', '-'),
}


def plot_hydrograph(
    obs: np.ndarray,
    preds: dict,
    basin_id: str,
    dates: pd.DatetimeIndex,
    out_path: str,
):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dates, obs, alpha=0.15, color='#444', label='Observed (fill)')
    ax.plot(dates, obs, color='#222', linewidth=1.2, label='Observed', zorder=5)
    for name, arr in preds.items():
        if name not in STYLES:
            continue
        color, ls = STYLES[name]
        ax.plot(dates, arr, color=color, linestyle=ls, linewidth=1.0, label=name)
    ax.set_ylabel('Streamflow (mm/day)')
    ax.set_title(f'Hydrograph — basin {basin_id} (test period)')
    ax.legend(ncol=6, fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_hydrograph_multi(hydro_pkl: str, out_path: str):
    """3-panel hydrograph (first/mid/last representative basins) from evaluate pickle."""
    if not Path(hydro_pkl).exists():
        print(f"Skipping hydrograph: {hydro_pkl} not found")
        return
    with open(hydro_pkl, "rb") as f:
        hydro_data = pickle.load(f)

    basins = list(hydro_data.keys())[:3]
    n = len(basins)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, bid in zip(axes, basins):
        data = hydro_data[bid]
        obs   = data["obs"]
        preds = data["preds"]
        dates = data.get("dates", range(len(obs)))

        ax.fill_between(dates, obs, alpha=0.12, color="#444")
        ax.plot(dates, obs, color="#222", linewidth=1.2, label="Observed", zorder=5)
        for name, arr in preds.items():
            if name not in STYLES:
                continue
            color, ls = STYLES[name]
            n_pts = min(len(obs), len(arr))
            ax.plot(dates[:n_pts], arr[:n_pts], color=color, linestyle=ls,
                    linewidth=0.9, label=name)
        ax.set_title(f"Basin {bid}", fontsize=10)
        ax.set_ylabel("Streamflow (mm/day)", fontsize=8)
        ax.legend(ncol=6, fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Date")
    plt.suptitle("Hydrograph Comparison — 3 Representative Basins", fontsize=11, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
