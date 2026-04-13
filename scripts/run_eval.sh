#!/usr/bin/env bash
# Evaluate the fine-tuned model on the test split.
# Usage: bash scripts/run_eval.sh [--config configs/base.yaml] [--split test]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/base.yaml}"
SPLIT="${2:-test}"

echo "========================================"
echo " AI Integrity Eval Lab — Evaluation"
echo " Config: $CONFIG  |  Split: $SPLIT"
echo "========================================"

python -m src.evaluate_model --config "$CONFIG" --split "$SPLIT"

echo ""
echo "Evaluation complete."
echo "  Metrics : outputs/metrics/eval_${SPLIT}_metrics.json"
echo "  Figures : outputs/figures/"
echo "  Misclassified: outputs/metrics/misclassified.csv"
