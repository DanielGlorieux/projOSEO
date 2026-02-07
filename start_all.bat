@echo off
echo ========================================
echo   ONEA Smart Energy Optimizer
echo   Demarrage du systeme complet
echo ========================================
echo.

:: Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe
    echo Telechargez Python depuis https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Verifier Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Node.js n'est pas installe
    echo Telechargez Node.js depuis https://nodejs.org/
    pause
    exit /b 1
)

:: Verifier .env
if not exist ".env" (
    echo [ATTENTION] Fichier .env non trouve
    echo Creez un fichier .env a partir de .env.example
    echo.
    pause
)

echo [1/3] Demarrage de l'API Backend (Port 8000)...
start "ONEA API Backend" cmd /k "python -m uvicorn api.main:app --reload --port 8000"

:: Attendre 5 secondes pour que l'API demarre
timeout /t 5 /nobreak >nul

echo [2/3] Demarrage du Dashboard Frontend (Port 3000)...
start "ONEA Dashboard Frontend" cmd /k "cd dashboard\react-app && npm run dev"

:: Attendre 3 secondes
timeout /t 3 /nobreak >nul

echo [3/3] Ouverture du navigateur...
timeout /t 10 /nobreak >nul
start http://localhost:3000

echo.
echo ========================================
echo   Systeme demarre avec succes !
echo ========================================
echo.
echo API Backend:     http://localhost:8000/docs
echo Dashboard:       http://localhost:3000
echo.
echo Appuyez sur Ctrl+C dans les fenetres pour arreter
echo ========================================
pause
