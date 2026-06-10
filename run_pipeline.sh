#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-../Caravan-csv.tar.xz}"
echo "=== Step 1: Extract Caravan archive ==="
python -c "from src.data.extract import extract_caravan; extract_caravan('$ARCHIVE')"

echo "=== Step 2: Preprocess ==="
python -m src.data.preprocess --output data/processed

echo "=== Step 3: Train RF ==="
python -m src.train.train_rf

echo "=== Step 4: Train XGBoost ==="
python -m src.train.train_xgb

echo "=== Step 5: Train GRU ==="
python -m src.train.train_dl --model gru

echo "=== Step 6: Train LSTM ==="
python -m src.train.train_dl --model lstm

echo "=== Step 7: Train ME-LSTM ==="
python -m src.train.train_dl --model me_lstm

echo "=== Step 8: Evaluate all models ==="
python -m src.evaluate.evaluate

echo "=== Step 9: Generate all plots ==="
python -m src.visualize.all_plots

echo "=== Pipeline complete. Results in results/ ==="
