#!/bin/bash

echo "========================================"
echo "    Crypto Filter Pro Launcher"
echo "========================================"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: .venv not found. Please create it first."
    exit 1
fi

function cleanup() {
    echo "Cleaning up any processes currently using port 8000 or 3000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    sleep 1
}

cleanup

echo "Starting Backend API on Port 3000..."
uvicorn main:app --port 3000 --reload &
API_PID=$!

echo "Starting Crypto Filter Shiny App on Port 8000..."
python -m shiny run app.py --reload --port 8000 --launch-browser

# Cleanup background process when Shiny exits
kill $API_PID
wait $API_PID 2>/dev/null
