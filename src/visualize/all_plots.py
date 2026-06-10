"""Orchestrate all visualizations."""
import argparse
from pathlib import Path

from src.utils.config import METRICS_DIR, PLOTS_DIR
from src.visualize.plot_skill_scores import plot_skill_scores
from src.visualize.plot_nse_distribution import plot_nse_distribution
from src.visualize.plot_flood_metrics import plot_flood_metrics
from src.visualize.plot_compute import plot_compute
from src.visualize.plot_radar import plot_radar
from src.visualize.plot_heatmap import plot_heatmap
from src.visualize.plot_convergence import plot_convergence
from src.visualize.plot_hydrograph import plot_hydrograph_multi


def run_all(results_csv: str, output_dir: str, metrics_dir: str = METRICS_DIR):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating skill score bars...")
    plot_skill_scores(results_csv, f"{output_dir}/nse_kge_bars.png")

    nse_pkl = Path(metrics_dir) / "nse_per_basin.pkl"
    if nse_pkl.exists():
        print("Generating NSE distribution box plot...")
        plot_nse_distribution(str(nse_pkl), f"{output_dir}/nse_distribution.png")

    print("Generating flood metrics...")
    plot_flood_metrics(results_csv, f"{output_dir}/flood_metrics.png")

    print("Generating compute efficiency plots...")
    plot_compute(results_csv, f"{output_dir}/compute_efficiency.png")

    print("Generating radar chart...")
    plot_radar(results_csv, f"{output_dir}/radar_all_models.png")

    print("Generating score heatmap...")
    plot_heatmap(results_csv, f"{output_dir}/score_heatmap.png")

    print("Generating convergence curves...")
    plot_convergence(metrics_dir, f"{output_dir}/convergence_curves.png")

    print("Generating hydrograph comparison...")
    hydro_pkl = Path(metrics_dir) / "hydrograph_data.pkl"
    plot_hydrograph_multi(str(hydro_pkl), f"{output_dir}/hydrograph_comparison.png")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    default=f"{METRICS_DIR}/summary.csv")
    parser.add_argument("--output",     default=PLOTS_DIR)
    parser.add_argument("--metrics_dir", default=METRICS_DIR)
    args = parser.parse_args()
    run_all(args.results, args.output, args.metrics_dir)
