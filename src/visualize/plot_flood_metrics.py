import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = ['RF', 'XGBoost', 'GRU', 'LSTM', 'ME-LSTM']
COLORS = ['#378ADD', '#BA7517', '#1D9E75', '#7F77DD', '#D85A30']


def plot_flood_metrics(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv).set_index('model')
    df = df.reindex([m for m in MODELS if m in df.index])
    models_present = df.index.tolist()
    colors_present = [COLORS[MODELS.index(m)] for m in models_present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.barh(models_present, df['MPTE_median'], color=colors_present,
            edgecolor=colors_present, linewidth=0.8)
    ax.set_xlabel('Days (lower = better)')
    ax.set_title('Mean Peak Timing Error (MPTE)')
    ax.axvline(0, color='#444', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    vals = df['PBIAS_HF_median']
    bar_colors = ['#E24B4A' if v > 0 else '#1D9E75' for v in vals]
    ax.bar(models_present, vals, color=bar_colors, edgecolor=bar_colors, linewidth=0.8)
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.set_ylabel('% bias')
    ax.set_title('PBIAS-HF — Top 2% Flows\n(red = overestimate, green = underestimate)')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
