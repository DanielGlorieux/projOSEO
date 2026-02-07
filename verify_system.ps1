# Script de vérification système ONEA
# Vérifie configuration, dépendances et état des services

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "VERIFICATION SYSTEME ONEA" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$errors = 0
$warnings = 0

# 1. Vérifier Python
Write-Host "[1/10] Verification Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   OK - $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ERREUR - Python non trouve" -ForegroundColor Red
    $errors++
}

# 2. Vérifier Node.js
Write-Host "`n[2/10] Verification Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "   OK - Node $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "   ERREUR - Node.js non trouve" -ForegroundColor Red
    $errors++
}

# 3. Vérifier dépendances Python
Write-Host "`n[3/10] Verification dependances Python..." -ForegroundColor Yellow
$pythonPackages = @("fastapi", "uvicorn", "pandas", "numpy", "torch", "langchain", "pinecone", "openpyxl")
foreach ($pkg in $pythonPackages) {
    try {
        $check = pip show $pkg 2>&1 | Select-String "Name:"
        if ($check) {
            Write-Host "   OK - $pkg installe" -ForegroundColor Green
        } else {
            Write-Host "   ATTENTION - $pkg non trouve" -ForegroundColor Yellow
            $warnings++
        }
    } catch {
        Write-Host "   ATTENTION - Impossible de verifier $pkg" -ForegroundColor Yellow
        $warnings++
    }
}

# 4. Vérifier .env
Write-Host "`n[4/10] Verification fichier .env..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "   OK - Fichier .env present" -ForegroundColor Green
    
    $envContent = Get-Content ".env" -Raw
    
    if ($envContent -match "GOOGLE_API_KEY=.+") {
        Write-Host "   OK - GOOGLE_API_KEY configuree" -ForegroundColor Green
    } else {
        Write-Host "   ATTENTION - GOOGLE_API_KEY manquante ou vide" -ForegroundColor Yellow
        $warnings++
    }
    
    if ($envContent -match "PINECONE_API_KEY=.+") {
        Write-Host "   OK - PINECONE_API_KEY configuree" -ForegroundColor Green
    } else {
        Write-Host "   ATTENTION - PINECONE_API_KEY manquante" -ForegroundColor Yellow
        $warnings++
    }
} else {
    Write-Host "   ERREUR - Fichier .env manquant" -ForegroundColor Red
    Write-Host "   Creer .env avec les cles API" -ForegroundColor Yellow
    $errors++
}

# 5. Vérifier structure dossiers
Write-Host "`n[5/10] Verification structure dossiers..." -ForegroundColor Yellow
$folders = @("api", "data\raw", "data\exports", "models", "dashboard\react-app")
foreach ($folder in $folders) {
    if (Test-Path $folder) {
        Write-Host "   OK - $folder" -ForegroundColor Green
    } else {
        Write-Host "   ATTENTION - $folder manquant" -ForegroundColor Yellow
        $warnings++
    }
}

# 6. Vérifier fichiers clés
Write-Host "`n[6/10] Verification fichiers cles..." -ForegroundColor Yellow
$files = @(
    "api\main.py",
    "api\chatbot_rag.py",
    "api\excel_export.py",
    "api\email_notifications.py",
    "dashboard\react-app\src\App.jsx",
    "dashboard\react-app\src\components\ChatbotAssistant.jsx",
    "dashboard\react-app\src\components\StationMap.jsx",
    "dashboard\react-app\src\components\EnergyChart.jsx"
)
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "   OK - $file" -ForegroundColor Green
    } else {
        Write-Host "   ERREUR - $file manquant" -ForegroundColor Red
        $errors++
    }
}

# 7. Vérifier données CSV
Write-Host "`n[7/10] Verification donnees CSV..." -ForegroundColor Yellow
$csvFiles = Get-ChildItem "data\raw\*.csv" -ErrorAction SilentlyContinue
if ($csvFiles) {
    Write-Host "   OK - $($csvFiles.Count) fichiers CSV trouves" -ForegroundColor Green
    foreach ($csv in $csvFiles | Select-Object -First 3) {
        $size = [math]::Round($csv.Length / 1MB, 2)
        Write-Host "      $($csv.Name) ($size MB)" -ForegroundColor Gray
    }
} else {
    Write-Host "   ERREUR - Aucun fichier CSV trouve" -ForegroundColor Red
    $errors++
}

# 8. Vérifier port 8000 (API)
Write-Host "`n[8/10] Verification API (port 8000)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "   OK - API accessible" -ForegroundColor Green
    Write-Host "   Status: $($response.StatusCode)" -ForegroundColor Gray
} catch {
    Write-Host "   ATTENTION - API non accessible" -ForegroundColor Yellow
    Write-Host "   Demarrer avec: python -m uvicorn api.main:app --reload --port 8000" -ForegroundColor Gray
    $warnings++
}

# 9. Vérifier port 3000 (Frontend)
Write-Host "`n[9/10] Verification Frontend (port 3000)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "   OK - Frontend accessible" -ForegroundColor Green
} catch {
    Write-Host "   ATTENTION - Frontend non accessible" -ForegroundColor Yellow
    Write-Host "   Demarrer avec: npm run dev (dans dashboard/react-app)" -ForegroundColor Gray
    $warnings++
}

# 10. Test endpoint critique
Write-Host "`n[10/10] Test endpoint critique..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/stations" -TimeoutSec 2 -ErrorAction Stop
    $stations = $response.Content | ConvertFrom-Json
    Write-Host "   OK - $($stations.Count) stations chargees" -ForegroundColor Green
} catch {
    Write-Host "   ATTENTION - Endpoint /stations inaccessible" -ForegroundColor Yellow
    $warnings++
}

# Résumé
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "RESUME" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "PARFAIT - Systeme pret!" -ForegroundColor Green
    Write-Host "`nPour demarrer:" -ForegroundColor Cyan
    Write-Host "  Terminal 1: python -m uvicorn api.main:app --reload --port 8000" -ForegroundColor Gray
    Write-Host "  Terminal 2: cd dashboard\react-app && npm run dev" -ForegroundColor Gray
    Write-Host "  Navigateur: http://localhost:3000`n" -ForegroundColor Gray
} elseif ($errors -eq 0) {
    Write-Host "ATTENTION - $warnings avertissements" -ForegroundColor Yellow
    Write-Host "Le systeme peut fonctionner avec limitations`n" -ForegroundColor Yellow
} else {
    Write-Host "ERREURS - $errors erreurs critiques, $warnings avertissements" -ForegroundColor Red
    Write-Host "Corriger les erreurs avant de demarrer`n" -ForegroundColor Red
}

Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  - START_RAPIDE.md (demarrage 3 min)" -ForegroundColor Gray
Write-Host "  - DEMARRAGE_COMPLET.md (guide detaille)" -ForegroundColor Gray
Write-Host "  - CORRECTIONS_BUGS.md (troubleshooting)" -ForegroundColor Gray
Write-Host "  - RESUME_FINAL_COMPLET.md (synthese complete)`n" -ForegroundColor Gray

Write-Host "Tests automatiques:" -ForegroundColor Cyan
Write-Host "  - test_systeme.bat (tests API complets)`n" -ForegroundColor Gray
