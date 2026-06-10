import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = ['RF', 'XGBoost', 'GRU', 'LSTM', 'ME-LSTM']
COLORS = ['#378ADD', '#BA7517', '#1D9E75', '#7F77DD', '#D85A30']
RADAR_METRICS = ['NSE', 'KGE', 'MPTE↓', 'PBIAS-HF↓', 'Latency↓', 'Memory↓']


def normalize_for_radar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ['NSE_median', 'KGE_median']:
        rng = df[col].max() - df[col].min()
        out[col] = (df[col] - df[col].min()) / (rng if rng > 0 else 1)
    for col in ['MPTE_median', 'PBIAS_HF_median', 'latency_ms', 'memory_MB']:
        rng = df[col].max() - df[col].min()
        out[col] = 1 - (df[col] - df[col].min()) / (rng if rng > 0 else 1)
    return out


def plot_radar(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv).set_index('model')
    df = df.reindex([m for m in MODELS if m in df.index])
    norm = normalize_for_radar(df)
    cols = ['NSE_median', 'KGE_median', 'MPTE_median', 'PBIAS_HF_median', 'latency_ms', 'memory_MB']

    N = len(cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    models_present = df.index.tolist()
    for i, m in enumerate(models_present):
        color = COLORS[MODELS.index(m)]
        vals = norm.loc[m, cols].tolist() + [norm.loc[m, cols[0]]]
        ax.plot(angles, vals, color=color, linewidth=1.5, label=m)
        ax.fill(angles, vals, color=color, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRICS, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7, color='gray')
    ax.set_title('Multi-metric radar (normalised)\nhigher = better on all axes', fontsize=10, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=False)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
