import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = ['RF', 'XGBoost', 'GRU', 'LSTM', 'ME-LSTM']
COLORS = ['#378ADD', '#BA7517', '#1D9E75', '#7F77DD', '#D85A30']


def plot_compute(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv).set_index('model')
    df = df.reindex([m for m in MODELS if m in df.index])
    models_present = df.index.tolist()
    colors_present = [COLORS[MODELS.index(m)] for m in models_present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.barh(models_present, df['latency_ms'], color=colors_present,
            edgecolor=colors_present, linewidth=0.8)
    ax.set_xlabel('Milliseconds (median, 100 runs)')
    ax.set_title('Inference latency')
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    max_conv = df['convergence_s'].max()
    sizes = (df['convergence_s'] / max(max_conv, 1)) * 800 + 80
    for i, m in enumerate(models_present):
        ax.scatter(df.loc[m, 'memory_MB'], df.loc[m, 'NSE_median'],
                   s=sizes[m], color=colors_present[i], edgecolors=colors_present[i],
                   alpha=0.8, linewidth=0.8)
        ax.annotate(m, (df.loc[m, 'memory_MB'], df.loc[m, 'NSE_median']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)
    ax.set_xlabel('Memory footprint (MB)')
    ax.set_ylabel('NSE (median)')
    ax.set_title('Memory vs. NSE\n(bubble area ∝ training time)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.annotate('← ideal', xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=8, color='gray')

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
