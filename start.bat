@echo off
echo ========================================
echo     Crypto Filter Pro Launcher (Windows)
echo ========================================

if not exist .venv (
    echo Error: .venv not found. Creating it...
    python -m venv .venv
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo Activating virtual environment...
    call .venv\Scripts\activate
)

echo Starting Crypto Filter Shiny App on Port 8000...
python -m shiny run app.py --reload --port 8000 --launch-browser
pause
