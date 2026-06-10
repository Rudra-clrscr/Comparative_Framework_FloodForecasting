import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODELS = ['RF', 'XGBoost', 'GRU', 'LSTM', 'ME-LSTM']
COLORS = ['#378ADD', '#BA7517', '#1D9E75', '#7F77DD', '#D85A30']


def plot_skill_scores(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv).set_index('model')
    df = df.reindex([m for m in MODELS if m in df.index])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, title in zip(
        axes,
        ['NSE_median', 'KGE_median'],
        ['NSE — Nash-Sutcliffe Efficiency', 'KGE — Kling-Gupta Efficiency']
    ):
        models_present = df.index.tolist()
        colors_present = [COLORS[MODELS.index(m)] for m in models_present]
        bars = ax.bar(models_present, df[col], color=colors_present,
                      edgecolor=colors_present, linewidth=0.8)
        iqr_col = col.replace('median', 'IQR')
        if iqr_col in df.columns:
            ax.errorbar(models_present, df[col], yerr=df[iqr_col] / 2,
                        fmt='none', color='#444', capsize=4, linewidth=1)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Acceptable (0.5)')
        ax.set_title(title, fontsize=12, fontweight='normal')
        ax.set_ylim(min(0, df[col].min() - 0.1), 1.05)
        ax.set_ylabel('Score (median ± IQR)')
        ax.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f'{val:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
