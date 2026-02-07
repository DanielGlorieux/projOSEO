@echo off
REM Script pour entrainer tous les modeles pour toutes les stations ONEA
REM Hackathon ONEA 2026

echo ========================================
echo ENTRAINEMENT COMPLET - TOUTES STATIONS
echo ========================================
echo.

REM Stations ONEA
set STATIONS=OUG_ZOG OUG_PIS BOBO_KUA OUG_NAB BOBO_DAR

echo Stations a entrainer: %STATIONS%
echo.
pause

REM Boucle sur chaque station
for %%S in (%STATIONS%) do (
    echo.
    echo ========================================
    echo STATION: %%S
    echo ========================================
    echo.
    
    python scripts\train_models.py --station %%S --models all
    
    if errorlevel 1 (
        echo [ERREUR] Echec entrainement station %%S
        pause
    ) else (
        echo [SUCCES] Station %%S entrainee!
    )
)

echo.
echo ========================================
echo ENTRAINEMENT TERMINE!
echo ========================================
echo.
echo Resultats disponibles dans: models/forecasting/ et models/optimization/
pause
