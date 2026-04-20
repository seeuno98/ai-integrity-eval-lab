#!/usr/bin/env bash
# Run the manual probe set against the saved model.
# Usage: bash scripts/run_probes.sh [configs/base.yaml]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/base.yaml}"

echo "========================================"
echo " AI Integrity Eval Lab — Probe Runner"
echo " Config: $CONFIG"
echo "========================================"

python -m src.run_probes --config "$CONFIG"

echo ""
echo "Probe outputs:"
echo "  Results : outputs/probes/probe_results.csv"
echo "  Summary : outputs/probes/probe_summary.json"
echo "  Report  : outputs/probes/probe_report.md"
