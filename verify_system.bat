@echo off
echo ╔══════════════════════════════════════════════════════════╗
echo ║  VERIFICATION COMPLETE SYSTEME ONEA                      ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM Se placer dans le bon répertoire
cd /d "%~dp0"

echo [1/7] Verification Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python non installe
    pause
    exit /b 1
)
echo ✅ Python installe

echo.
echo [2/7] Verification Node.js...
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js non installe
    pause
    exit /b 1
)
echo ✅ Node.js installe

echo.
echo [3/7] Verification fichier .env...
if not exist ".env" (
    echo ❌ Fichier .env manquant
    echo.
    echo Creez un fichier .env avec:
    echo GOOGLE_API_KEY=votre_cle
    echo PINECONE_API_KEY=votre_cle
    echo PINECONE_INDEX_NAME=onea-knowledge-base
    pause
    exit /b 1
)
echo ✅ Fichier .env present

echo.
echo [4/7] Verification donnees CSV...
set CSV_COUNT=0
for %%f in (data\raw\*_historical.csv) do set /a CSV_COUNT+=1
if %CSV_COUNT% EQU 0 (
    echo ⚠️ Aucun fichier CSV trouve dans data\raw\
    echo    Export Excel ne fonctionnera pas
) else (
    echo ✅ %CSV_COUNT% fichiers CSV trouves
)

echo.
echo [5/7] Verification packages Python...
pip show fastapi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Packages Python manquants
    echo    Executez: pip install -r requirements.txt
    pause
    exit /b 1
)
echo ✅ Packages Python installes

echo.
echo [6/7] Verification node_modules React...
if not exist "dashboard\react-app\node_modules" (
    echo ⚠️ node_modules manquant
    echo    Executez: cd dashboard\react-app ^&^& npm install
    set NEED_NPM_INSTALL=1
) else (
    echo ✅ node_modules present
    set NEED_NPM_INSTALL=0
)

echo.
echo [7/7] Verification structure dossiers...
set ALL_OK=1

if not exist "api\" (
    echo ❌ Dossier api\ manquant
    set ALL_OK=0
)
if not exist "models\" (
    echo ❌ Dossier models\ manquant
    set ALL_OK=0
)
if not exist "data\" (
    echo ❌ Dossier data\ manquant
    set ALL_OK=0
)
if not exist "dashboard\react-app\src\" (
    echo ❌ Dossier dashboard\react-app\src\ manquant
    set ALL_OK=0
)

if %ALL_OK% EQU 1 (
    echo ✅ Structure dossiers correcte
) else (
    echo ❌ Structure dossiers incomplete
    pause
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  RESUME VERIFICATION                                     ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

if %NEED_NPM_INSTALL% EQU 1 (
    echo ⚠️ Action requise: Installer node_modules
    echo    cd dashboard\react-app
    echo    npm install
    echo.
)

echo ✅ Systeme pret pour demarrage
echo.
echo Pour lancer le systeme:
echo   1. Terminal 1: start_api.bat
echo   2. Terminal 2: start_dashboard.bat
echo.
echo Documentation:
echo   - GUIDE_DEMARRAGE.md
echo   - MODELES_IA_DOCUMENTATION.md
echo   - CORRECTIONS_FINALES.md
echo   - SYNTHESE_PRESENTATION.md
echo.

pause
