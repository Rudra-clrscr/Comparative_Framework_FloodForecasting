"""Windows-friendly pipeline runner."""
import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable


def run(cmd: list, **kwargs):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="../Caravan-csv.tar.xz")
    parser.add_argument("--skip_extract",    action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train",      action="store_true")
    parser.add_argument("--skip_evaluate",   action="store_true")
    parser.add_argument("--skip_plots",      action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent

    if not args.skip_extract:
        print("=== Step 1: Extract ===")
        run([PYTHON, "-c",
             f"import sys; sys.path.insert(0,'.'); from src.data.extract import extract_caravan; extract_caravan('{args.archive}', 'data/raw')"],
            cwd=base)

    if not args.skip_preprocess:
        print("=== Step 2: Preprocess ===")
        run([PYTHON, "-m", "src.data.preprocess"], cwd=base)

    if not args.skip_train:
        print("=== Step 3: Train RF ===")
        run([PYTHON, "-m", "src.train.train_rf"], cwd=base)

        print("=== Step 4: Train XGBoost ===")
        run([PYTHON, "-m", "src.train.train_xgb"], cwd=base)

        for model in ["gru", "lstm", "me_lstm"]:
            print(f"=== Step 5: Train {model.upper()} ===")
            run([PYTHON, "-m", "src.train.train_dl", "--model", model], cwd=base)

    if not args.skip_evaluate:
        print("=== Step 6: Evaluate ===")
        run([PYTHON, "-m", "src.evaluate.evaluate"], cwd=base)

    if not args.skip_plots:
        print("=== Step 7: Plots ===")
        run([PYTHON, "-m", "src.visualize.all_plots"], cwd=base)

    print("\n=== Pipeline complete! ===")


if __name__ == "__main__":
    main()
