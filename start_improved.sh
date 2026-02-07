#!/bin/bash
echo "========================================"
echo "ONEA Smart Energy Optimizer - AMÉLIORE"
echo "========================================"
echo ""

echo "[1/3] Vérification des dépendances Python..."
pip install -q xgboost lightgbm optuna scikit-learn pandas numpy fastapi uvicorn

echo ""
echo "[2/3] Démarrage API FastAPI..."
cd api
python main.py &
API_PID=$!
cd ..
sleep 5

echo ""
echo "[3/3] Démarrage Dashboard React..."
cd dashboard/react-app
npm run dev &
DASHBOARD_PID=$!
cd ../..

echo ""
echo "========================================"
echo "Services démarrés!"
echo "========================================"
echo "API:       http://localhost:8000"
echo "Dashboard: http://localhost:3000"
echo "API Docs:  http://localhost:8000/docs"
echo "========================================"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter..."

# Trap SIGINT to kill background processes
trap "kill $API_PID $DASHBOARD_PID; exit" INT

# Wait for background processes
wait
