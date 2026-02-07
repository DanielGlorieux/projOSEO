"""
Script d'entrainement rapide des modeles optimises
"""
import sys
import os
from pathlib import Path
import pandas as pd
import time

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(str(Path(__file__).parent.parent))

from models.forecasting.ensemble_model import EnsembleForecastModel
from models.optimization.smart_optimizer import SmartPumpOptimizer

def main():
    print("=" * 80)
    print("ONEA Smart Energy Optimizer - Entrainement Modeles Optimises")
    print("=" * 80)
    
    data_path = Path(__file__).parent.parent / "data" / "raw"
    models_path = Path(__file__).parent.parent / "models"
    
    stations = [
        {
            'id': 'OUG_ZOG',
            'name': 'Station de Zogona',
            'pump_count': 4,
            'max_power_kw': 850,
            'max_flow_m3h': 3500
        },
        {
            'id': 'OUG_PIS',
            'name': 'Station de Pissy',
            'pump_count': 6,
            'max_power_kw': 1200,
            'max_flow_m3h': 5000
        },
        {
            'id': 'BOBO_KUA',
            'name': 'Station de Kuinima',
            'pump_count': 5,
            'max_power_kw': 950,
            'max_flow_m3h': 4000
        },
        {
            'id': 'OUG_NAB',
            'name': 'Station de Nabitenga',
            'pump_count': 3,
            'max_power_kw': 600,
            'max_flow_m3h': 2500
        },
        {
            'id': 'BOBO_DAR',
            'name': 'Station de Darsalamy',
            'pump_count': 4,
            'max_power_kw': 750,
            'max_flow_m3h': 3200
        }
    ]
    
    all_results = []
    total_start = time.time()
    
    for station in stations:
        station_id = station['id']
        csv_file = data_path / f"{station_id}_historical.csv"
        
        if not csv_file.exists():
            print(f"\nDonnees non trouvees: {station_id}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Station: {station['name']} ({station_id})")
        print('='*80)
        
        start_time = time.time()
        
        # Charger données
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        print(f"Donnees chargees: {len(df)} enregistrements")
        
        # Entrainement modèle de prédiction
        print("\nMODELE DE PREDICTION (XGBoost + LightGBM)")
        print("-" * 80)
        
        forecast_model = EnsembleForecastModel()
        metrics = forecast_model.train(df)
        
        # Sauvegarde
        model_file = models_path / "forecasting" / f"{station_id}_ensemble.joblib"
        forecast_model.save(model_file)
        
        # Test prédiction
        predictions = forecast_model.predict(df, horizon_hours=24)
        print(f"\n🔮 Test Prédictions 24h:")
        print(f"   Min:     {predictions.min():.1f} m³/h")
        print(f"   Max:     {predictions.max():.1f} m³/h")
        print(f"   Moyenne: {predictions.mean():.1f} m³/h")
        
        # Test optimiseur
        print("\nOPTIMISEUR INTELLIGENT (Optuna)")
        print("-" * 80)
        
        optimizer = SmartPumpOptimizer({
            'station_id': station_id,
            'name': station['name'],
            'pump_count': station['pump_count'],
            'max_power_kw': station['max_power_kw'],
            'max_flow_m3h': station['max_flow_m3h']
        })
        
        # Test sur demande moyenne
        avg_demand = df['water_demand_m3'].mean()
        result = optimizer.optimize_for_period(12, avg_demand)
        
        print(f"   Demande test:      {avg_demand:.0f} m³/h")
        print(f"   Pompes optimales:  {result.pumps_active}/{station['pump_count']}")
        print(f"   Puissance:         {result.power_level*100:.0f}%")
        print(f"   Économie prévue:   {result.expected_savings:.0f} FCFA ({result.savings_percent:.1f}%)")
        
        elapsed = time.time() - start_time
        
        all_results.append({
            'station': station['name'],
            'mape': metrics['mape'],
            'r2': metrics['r2'],
            'savings_percent': result.savings_percent,
            'training_time': elapsed
        })
        
        print(f"\nTemps d'entrainement: {elapsed:.1f}s")
    
    # Résumé global
    total_time = time.time() - total_start
    
    print("\n" + "=" * 80)
    print("RESUME GLOBAL")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    print(f"\nEntrainement termine en {total_time:.1f}s")
    print(f"MAPE moyen: {results_df['mape'].mean():.2f}%")
    print(f"R2 moyen: {results_df['r2'].mean():.4f}")
    print(f"Economies moyennes: {results_df['savings_percent'].mean():.1f}%")
    print("\nModeles prets pour production!")


if __name__ == "__main__":
    main()
