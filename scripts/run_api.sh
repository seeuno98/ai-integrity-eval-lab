#!/usr/bin/env bash
# Start the FastAPI inference server.
# Usage: bash scripts/run_api.sh [port]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PORT="${1:-8000}"

echo "========================================"
echo " AI Integrity Eval Lab — API Server"
echo " Listening on http://0.0.0.0:$PORT"
echo " Docs at      http://localhost:$PORT/docs"
echo "========================================"

uvicorn src.api:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
