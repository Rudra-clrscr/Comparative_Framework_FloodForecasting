import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_convergence(history_dir: str, out_path: str):
    DL_MODELS = ['GRU', 'LSTM', 'ME-LSTM']
    styles = {
        'GRU':     ('#1D9E75', '-'),
        'LSTM':    ('#7F77DD', '--'),
        'ME-LSTM': ('#D85A30', '-.'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    any_plotted = False

    for m in DL_MODELS:
        hist_path = Path(history_dir) / f"{m.lower()}_history.csv"
        if not hist_path.exists():
            continue
        hist = pd.read_csv(hist_path)
        color, ls = styles[m]
        axes[0].plot(hist['epoch'], hist['train_loss'], color=color, linestyle=ls, label=m)
        axes[1].plot(hist['epoch'], hist['val_nse'],   color=color, linestyle=ls, label=m)
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    for ax, title, ylabel in zip(
        axes,
        ['Training loss (MSE, normalised scale)', 'Validation NSE per epoch'],
        ['MSE loss', 'NSE'],
    ):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
