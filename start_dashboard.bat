@echo off
echo ========================================
echo   DEMARRAGE DASHBOARD REACT
echo ========================================
echo.

REM Se placer dans le dossier React
cd /d "%~dp0\dashboard\react-app"

echo Verification de l'installation de node_modules...
if not exist "node_modules" (
    echo Installation des dependances npm...
    call npm install
)

echo.
echo Demarrage du dashboard sur http://localhost:3000
echo.

REM Demarrer React
call npm run dev

pause
