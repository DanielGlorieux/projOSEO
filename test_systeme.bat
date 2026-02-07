@echo off
REM Script de test automatique - ONEA Smart Energy Optimizer
REM Teste tous les endpoints et fonctionnalités

echo ====================================
echo TESTS ONEA SMART ENERGY OPTIMIZER
echo ====================================
echo.

REM Couleurs
echo [93mVerification de l'API...[0m
echo.

REM Test 1: Health Check
echo [96m1. Test Health Check[0m
curl -s http://localhost:8000/health
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: API non accessible[0m
    echo [93m   Solution: Demarrer l'API avec start_api.bat[0m
    pause
    exit /b 1
)
echo [92m   OK - API accessible[0m
echo.

REM Test 2: Liste stations
echo [96m2. Test Liste Stations[0m
curl -s http://localhost:8000/stations > test_stations.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Impossible de charger les stations[0m
    pause
    exit /b 1
)
echo [92m   OK - Stations chargees[0m
type test_stations.json | findstr "OUG_ZOG"
echo.

REM Test 3: Details station
echo [96m3. Test Details Station OUG_ZOG[0m
curl -s http://localhost:8000/station/OUG_ZOG > test_station_details.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Endpoint /station/{id} non trouve[0m
    pause
    exit /b 1
)
echo [92m   OK - Details station recuperes[0m
type test_station_details.json | findstr "efficiency"
echo.

REM Test 4: Forecast
echo [96m4. Test Predictions 24h[0m
curl -s -X POST http://localhost:8000/forecast ^
  -H "Content-Type: application/json" ^
  -d "{\"station_id\":\"OUG_ZOG\",\"horizon_hours\":24}" ^
  > test_forecast.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Predictions echouees[0m
    pause
    exit /b 1
)
echo [92m   OK - Predictions 24h generees[0m
type test_forecast.json | findstr "predictions"
echo.

REM Test 5: Analyse horaire
echo [96m5. Test Analyse Horaire Efficacite[0m
curl -s "http://localhost:8000/analytics/hourly-efficiency/OUG_ZOG?days=7" > test_hourly.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Analyse horaire echouee[0m
    pause
    exit /b 1
)
echo [92m   OK - Analyse horaire generee[0m
type test_hourly.json | findstr "hourly_data"
echo.

REM Test 6: Optimisation
echo [96m6. Test Optimisation Pompage[0m
curl -s -X POST http://localhost:8000/optimize ^
  -H "Content-Type: application/json" ^
  -d "{\"station_id\":\"OUG_ZOG\",\"current_state\":{}}" ^
  > test_optimize.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Optimisation echouee[0m
    pause
    exit /b 1
)
echo [92m   OK - Recommandations generees[0m
type test_optimize.json | findstr "recommended_actions"
echo.

REM Test 7: Chatbot suggestions
echo [96m7. Test Chatbot Suggestions[0m
curl -s http://localhost:8000/chatbot/suggestions > test_chatbot_suggestions.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Suggestions chatbot echouees[0m
    echo [93m   Info: Normal si RAG non configure[0m
) else (
    echo [92m   OK - Suggestions chatbot chargees[0m
    type test_chatbot_suggestions.json | findstr "question"
)
echo.

REM Test 8: Chatbot query
echo [96m8. Test Chatbot Query[0m
curl -s -X POST http://localhost:8000/chatbot/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Comment reduire la consommation energetique?\"}" ^
  > test_chatbot_query.json
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Query chatbot echouee[0m
    echo [93m   Info: Normal si RAG non configure[0m
) else (
    echo [92m   OK - Reponse chatbot generee[0m
    type test_chatbot_query.json | findstr "answer"
)
echo.

REM Test 9: Export Excel
echo [96m9. Test Export Excel[0m
curl -s -X POST http://localhost:8000/export/station-data ^
  -H "Content-Type: application/json" ^
  -d "{\"station_id\":\"OUG_ZOG\",\"days\":7}" ^
  --output test_export.xlsx
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Export Excel echoue[0m
    echo [93m   Info: Normal si service export non disponible[0m
) else (
    if exist test_export.xlsx (
        echo [92m   OK - Fichier Excel genere[0m
        for %%A in (test_export.xlsx) do echo    Taille: %%~zA octets
        if %%~zA LSS 1000 (
            echo [93m   ATTENTION: Fichier trop petit, possiblement vide[0m
        )
    ) else (
        echo [91m   ERREUR: Fichier Excel non cree[0m
    )
)
echo.

REM Test 10: Frontend accessible
echo [96m10. Test Frontend React[0m
curl -s http://localhost:3000 > test_frontend.html
if %ERRORLEVEL% NEQ 0 (
    echo [91m   ERREUR: Frontend non accessible[0m
    echo [93m   Solution: Demarrer avec start_dashboard.bat[0m
) else (
    echo [92m   OK - Frontend accessible[0m
)
echo.

echo ====================================
echo RESUME DES TESTS
echo ====================================
echo.

REM Compter les fichiers de test crees
set count=0
for %%F in (test_*.json test_*.xlsx test_*.html) do set /a count+=1

echo [92mFichiers de test generes: %count%[0m
echo.

echo [96mFichiers crees:[0m
dir /b test_*.* 2>nul
echo.

echo [93mPour nettoyer les fichiers de test:[0m
echo   del test_*.json test_*.xlsx test_*.html
echo.

echo ====================================
echo VERIFICATION STRUCTURE
echo ====================================
echo.

REM Verifier fichiers importants
echo [96mVerification fichiers cles:[0m

if exist "api\main.py" (
    echo [92m   OK api\main.py[0m
) else (
    echo [91m   MANQUANT api\main.py[0m
)

if exist "api\chatbot_rag.py" (
    echo [92m   OK api\chatbot_rag.py[0m
) else (
    echo [91m   MANQUANT api\chatbot_rag.py[0m
)

if exist "api\excel_export.py" (
    echo [92m   OK api\excel_export.py[0m
) else (
    echo [91m   MANQUANT api\excel_export.py[0m
)

if exist "api\email_notifications.py" (
    echo [92m   OK api\email_notifications.py[0m
) else (
    echo [91m   MANQUANT api\email_notifications.py[0m
)

if exist ".env" (
    echo [92m   OK .env[0m
    echo [93m   Verification cles API...[0m
    findstr "GOOGLE_API_KEY" .env >nul
    if %ERRORLEVEL% EQU 0 (
        echo [92m      OK GOOGLE_API_KEY configuree[0m
    ) else (
        echo [93m      ATTENTION: GOOGLE_API_KEY non trouvee[0m
    )
    findstr "PINECONE_API_KEY" .env >nul
    if %ERRORLEVEL% EQU 0 (
        echo [92m      OK PINECONE_API_KEY configuree[0m
    ) else (
        echo [93m      ATTENTION: PINECONE_API_KEY non trouvee[0m
    )
) else (
    echo [91m   MANQUANT .env[0m
    echo [93m   Creer .env avec les cles API[0m
)

if exist "data\raw\OUG_ZOG_historical.csv" (
    echo [92m   OK Donnees CSV presentes[0m
) else (
    echo [91m   MANQUANT Donnees CSV[0m
)

if exist "dashboard\react-app\src\components\ChatbotAssistant.jsx" (
    echo [92m   OK ChatbotAssistant.jsx[0m
) else (
    echo [91m   MANQUANT ChatbotAssistant.jsx[0m
)

if exist "dashboard\react-app\src\components\StationMap.jsx" (
    echo [92m   OK StationMap.jsx[0m
) else (
    echo [93m   ATTENTION: StationMap.jsx non trouve[0m
)

echo.

echo ====================================
echo TESTS TERMINES
echo ====================================
echo.

echo [96mPour demarrer le systeme complet:[0m
echo   1. Terminal 1: start_api.bat
echo   2. Terminal 2: start_dashboard.bat
echo   3. Ouvrir http://localhost:3000
echo.

echo [96mDocumentation:[0m
echo   - DEMARRAGE_COMPLET.md
echo   - CORRECTIONS_BUGS.md
echo   - RESUME_FINAL_COMPLET.md
echo   - API docs: http://localhost:8000/docs
echo.

pause
