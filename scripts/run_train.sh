#!/usr/bin/env bash
# Train the toxicity classifier.
# Usage: bash scripts/run_train.sh [--config configs/base.yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/base.yaml}"

echo "========================================"
echo " AI Integrity Eval Lab — Training"
echo " Config: $CONFIG"
echo "========================================"

python -m src.train --config "$CONFIG"

echo ""
echo "Training complete. Artifacts saved to outputs/."
