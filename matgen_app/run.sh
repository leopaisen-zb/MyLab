#!/usr/bin/env bash
# Launch MatGen-Eq Streamlit app
# Usage: bash run.sh [port]

PORT=${1:-8501}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting RAG + Eq-Light system on port $PORT ..."
streamlit run "$SCRIPT_DIR/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
