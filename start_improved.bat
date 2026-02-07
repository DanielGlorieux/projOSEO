@echo off
echo ========================================
echo ONEA Smart Energy Optimizer - AMELIORE
echo ========================================
echo.

echo [1/3] Verification des dependances Python...
pip install -q xgboost lightgbm optuna scikit-learn pandas numpy fastapi uvicorn

echo.
echo [2/3] Demarrage API FastAPI...
start "ONEA API" cmd /k "cd api && python main.py"
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Demarrage Dashboard React...
start "ONEA Dashboard" cmd /k "cd dashboard\react-app && npm run dev"

echo.
echo ========================================
echo Services demarres!
echo ========================================
echo API:       http://localhost:8000
echo Dashboard: http://localhost:3000
echo API Docs:  http://localhost:8000/docs
echo ========================================
echo.
echo Appuyez sur une touche pour voir les logs...
pause >nul
