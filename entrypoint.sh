#!/bin/bash
set -e

PORT="${SHINY_PORT:-8000}"

echo "=========================================="
echo "    Crypto Filter Shiny Launcher"
echo "=========================================="

# Start internal FastAPI backend on port 3000
echo "[1/2] Starting Backend API on 127.0.0.1:3000..."
uvicorn main:app --host 127.0.0.1 --port 3000 &
API_PID=$!

# Graceful cleanup handler
cleanup() {
    echo "Stopping Crypto Filter..."
    kill -TERM "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for API to be ready before starting Shiny
echo "Waiting for Backend API to become ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:3000/ > /dev/null 2>&1; then
        echo "Backend API is ready."
        break
    fi
    sleep 0.5
done

# Start Shiny App
echo "[2/2] Starting Crypto Filter Shiny App on 0.0.0.0:${PORT}..."
python -m shiny run app.py --host 0.0.0.0 --port "${PORT}" &
SHINY_PID=$!

# Wait for Shiny process
wait $SHINY_PID
cleanup
