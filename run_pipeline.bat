@echo off
title Diabetes Prediction ML Pipeline Runner
echo ====================================================
echo   Diabetes Prediction — ML Pipeline Runner
echo ====================================================
echo [1/2] Installing requirements...
pip install -r requirements.txt
echo.
echo [2/2] Training models and generating results...
python main.py
echo.
echo ML pipeline execution completed!
pause
