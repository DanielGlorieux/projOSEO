@echo off
echo ========================================
echo   DEMARRAGE API ONEA
echo ========================================
echo.

REM Se placer dans le bon répertoire
cd /d "%~dp0"

echo Verification des variables d'environnement...
if not exist ".env" (
    echo ERREUR: Fichier .env manquant!
    echo Creez un fichier .env avec les cles API necessaires:
    echo   GOOGLE_API_KEY=votre_cle_google
    echo   PINECONE_API_KEY=votre_cle_pinecone
    echo   PINECONE_INDEX_NAME=onea-knowledge-base
    pause
    exit /b 1
)

echo.
echo Demarrage de l'API sur http://localhost:8000
echo.
echo Documentation API: http://localhost:8000/docs
echo.

REM Demarrer l'API avec uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

pause
