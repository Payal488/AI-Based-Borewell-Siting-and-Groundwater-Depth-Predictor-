@echo off
title Borewell Success Predictor - B.Tech Capstone MVP
color 0b
echo ========================================================
echo   AI-Based Groundwater Depletion & Borewell Predictor
echo   B.Tech Capstone Project MVP - Windows Launcher
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked.
    pause
    exit /b
)

if not exist "venv\" (
    echo [*] Creating virtual environment...
    python -m venv venv
)

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Checking dependencies...
pip install -r requirements.txt --quiet

if not exist "models\depth_regressor.joblib" (
    echo [*] Training XGBoost models...
    python src\train_models.py
)

echo.
echo ========================================================
echo   Launching Streamlit Dashboard...
echo   Open http://localhost:8501 in your browser.
echo ========================================================
streamlit run app.py

pause
