import pickle
import matplotlib.pyplot as plt
from pathlib import Path

MODELS = ['RF', 'XGBoost', 'GRU', 'LSTM', 'ME-LSTM']
COLORS = ['#378ADD', '#BA7517', '#1D9E75', '#7F77DD', '#D85A30']


def plot_nse_distribution(per_basin_pkl: str, out_path: str):
    with open(per_basin_pkl, 'rb') as f:
        data = pickle.load(f)

    models_present = [m for m in MODELS if m in data and len(data[m]) > 0]
    colors_present = [COLORS[MODELS.index(m)] for m in models_present]

    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot(
        [data[m] for m in models_present],
        labels=models_present,
        patch_artist=True,
        notch=False,
        medianprops=dict(color='white', linewidth=2),
    )
    for patch, color in zip(bp['boxes'], colors_present):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='NSE=0.5')
    ax.set_ylabel('NSE')
    ax.set_title('NSE distribution across basins')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
