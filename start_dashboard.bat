@echo off
title Diabetes Prediction Dashboard Server
echo ====================================================
echo   Starting Diabetes Prediction Dashboard
echo ====================================================
echo Opening http://localhost:8080/ in your browser...
start "" http://localhost:8080/
python -m http.server 8080 --directory dashboard
pause
