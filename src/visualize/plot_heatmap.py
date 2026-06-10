import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.visualize.plot_radar import normalize_for_radar, MODELS


def plot_heatmap(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv).set_index('model')
    df = df.reindex([m for m in MODELS if m in df.index])
    norm = normalize_for_radar(df)
    cols = ['NSE_median', 'KGE_median', 'MPTE_median', 'PBIAS_HF_median', 'latency_ms', 'memory_MB']
    heat = norm[cols].rename(columns={
        'NSE_median':    'NSE',
        'KGE_median':    'KGE',
        'MPTE_median':   'MPTE↓',
        'PBIAS_HF_median': 'PBIAS-HF↓',
        'latency_ms':    'Latency↓',
        'memory_MB':     'Memory↓',
    })

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(heat, annot=True, fmt='.2f', cmap='YlGn',
                vmin=0, vmax=1, linewidths=0.4, ax=ax,
                cbar_kws={'label': 'Normalised score (higher = better)'})
    ax.set_title('Score heatmap — all models × all metrics')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
