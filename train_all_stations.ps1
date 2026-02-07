# Script PowerShell pour entrainer tous les modeles pour toutes les stations ONEA
# Hackathon ONEA 2026

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENTRAINEMENT COMPLET - TOUTES STATIONS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Stations ONEA
$stations = @("OUG_ZOG", "OUG_PIS", "BOBO_KUA", "OUG_NAB", "BOBO_DAR")

Write-Host "Stations à entraîner: $($stations -join ', ')" -ForegroundColor Yellow
Write-Host ""
Write-Host "Appuyez sur une touche pour commencer..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

$successCount = 0
$errorCount = 0
$startTime = Get-Date

# Boucle sur chaque station
foreach ($station in $stations) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "STATION: $station" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    
    try {
        python scripts\train_models.py --station $station --models all
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[✓ SUCCES] Station $station entrainee!" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "[✗ ERREUR] Echec entrainement station $station" -ForegroundColor Red
            $errorCount++
        }
    }
    catch {
        Write-Host "[✗ ERREUR] Exception: $_" -ForegroundColor Red
        $errorCount++
    }
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENTRAINEMENT TERMINE!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Résumé:" -ForegroundColor Yellow
Write-Host "  ✓ Succès: $successCount stations" -ForegroundColor Green
Write-Host "  ✗ Erreurs: $errorCount stations" -ForegroundColor Red
Write-Host "  ⏱ Durée totale: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Yellow
Write-Host ""
Write-Host "Résultats disponibles dans:" -ForegroundColor Cyan
Write-Host "  - models/forecasting/" -ForegroundColor White
Write-Host "  - models/optimization/" -ForegroundColor White
Write-Host ""
pause
