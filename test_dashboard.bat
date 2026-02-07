@echo off
REM Script de test rapide du dashboard ameliore

echo.
echo ================================
echo ONEA Dashboard Test
echo ================================
echo.

cd dashboard\react-app

echo Installation des dependances...
call npm install

echo.
echo Build du dashboard...
call npm run build

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================
    echo ✓ Build reussi!
    echo ================================
    echo.
    echo Pour lancer le dashboard en mode developpement:
    echo   cd dashboard\react-app
    echo   npm run dev
    echo.
    echo Le dashboard sera accessible sur http://localhost:3000
    echo.
    echo Nouvelles fonctionnalites:
    echo   - Systeme d'onglets moderne avec Radix UI
    echo   - Cards animees avec shadcn/ui
    echo   - Transitions fluides et effets de hover
    echo   - Footer identique a FrontendDashboard
    echo   - Design responsive et professionnel
    echo.
) else (
    echo.
    echo ================================
    echo X Erreur lors du build
    echo ================================
    exit /b 1
)

cd ..\..
