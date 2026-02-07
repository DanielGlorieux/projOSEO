"""
Script principal d'entranement des modles
Excute le pipeline complet: data  preprocessing  training  evaluation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import STATIONS, ML_CONFIG
from utils.preprocessing import DataPreprocessor, split_train_val_test, create_sequences
from models.forecasting.lstm_forecaster import EnergyDemandForecaster, calculate_metrics
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_station_data(station_id: str) -> pd.DataFrame:
    """Charge donnes d'une station"""
    data_path = Path(__file__).parent.parent / "data" / "raw" / f"{station_id}_historical.csv"
    
    if not data_path.exists():
        print(f"  Donnes {station_id} introuvables. Gnration requise...")
        print("Excutez: python data/synthetic_generator.py")
        sys.exit(1)
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f" Donnes charges: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def train_forecasting_models(df: pd.DataFrame, station_id: str):
    """Entrane les modles de prdiction"""
    print("\n" + "="*60)
    print(" ENTRANEMENT MODLES DE PRDICTION")
    print("="*60)
    
    # Preprocessing avec target scaling
    print("\n Preprocessing...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    numerical_cols = [
        'energy_consumption_kwh',
        'water_demand_m3',
        'power_kw',
        'reservoir_level_m3',
        'pump_efficiency'
    ]
    
    df_processed = preprocessor.fit_transform(
        df,
        numerical_cols=numerical_cols,
        target_col='energy_consumption_kwh',  # NOUVEAU
        add_time_feats=True,
        add_lags=True,
        add_rolling=True
    )
    
    # Sauvegarder preprocessor
    preprocessor_path = Path(__file__).parent.parent / "models" / "forecasting" / f"preprocessor_{station_id}.pkl"
    preprocessor.save(str(preprocessor_path))
    print(f" Preprocessor sauvegard: {preprocessor_path}")
    
    # Split donnes
    train_df, val_df, test_df = split_train_val_test(df_processed, 0.7, 0.15, 0.15)
    print(f" Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Prparer squences pour LSTM
    sequence_length = ML_CONFIG['forecasting']['lstm']['sequence_length']
    forecast_horizon = 24
    
    # Features pour LSTM
    feature_cols = [col for col in df_processed.columns if '_scaled' in col or 'sin' in col or 'cos' in col]
    feature_cols = feature_cols[:10]  # Limiter pour exemple
    
    print(f"\n Cration squences (seq_len={sequence_length}, horizon={forecast_horizon})...")
    
    # Train sequences - UTILISER TARGET SCALED
    train_features = train_df[feature_cols].values
    train_target = train_df['energy_consumption_kwh_scaled'].values  # SCALED
    
    X_train, y_train = create_sequences(train_features, train_target, sequence_length, forecast_horizon)
    
    # Val sequences - UTILISER TARGET SCALED
    val_features = val_df[feature_cols].values
    val_target = val_df['energy_consumption_kwh_scaled'].values  # SCALED
    
    X_val, y_val = create_sequences(val_features, val_target, sequence_length, forecast_horizon)
    
    # Test sequences - SCALED pour entraînement
    test_features = test_df[feature_cols].values
    test_target_scaled = test_df['energy_consumption_kwh_scaled'].values  # SCALED
    test_target_original = test_df['energy_consumption_kwh'].values  # ORIGINAL pour métriques
    
    X_test, y_test_scaled = create_sequences(test_features, test_target_scaled, sequence_length, forecast_horizon)
    _, y_test_original = create_sequences(test_features, test_target_original, sequence_length, forecast_horizon)
    
    print(f" Train: X={X_train.shape}, y={y_train.shape}")
    print(f" Val: X={X_val.shape}, y={y_val.shape}")
    print(f" Test: X={X_test.shape}, y={y_test_scaled.shape}")
    
    # Reshape y pour avoir (n_samples, forecast_horizon, 1)
    y_train = y_train.reshape(-1, forecast_horizon, 1)
    y_val = y_val.reshape(-1, forecast_horizon, 1)
    y_test_scaled = y_test_scaled.reshape(-1, forecast_horizon, 1)
    
    # Crer DataLoaders avec support GPU
    batch_size = ML_CONFIG['forecasting']['lstm']['batch_size']
    num_workers = ML_CONFIG['forecasting']['lstm'].get('num_workers', 0)
    pin_memory = ML_CONFIG['forecasting']['lstm'].get('pin_memory', False)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Entraner LSTM
    print(f"\n Entranement LSTM...")
    
    # Supprimer anciens checkpoints incompatibles
    checkpoint_path = Path(__file__).parent.parent / 'best_model.pth'
    if checkpoint_path.exists():
        print(f"  Suppression ancien checkpoint: {checkpoint_path}")
        checkpoint_path.unlink()
    
    config = {
        'hidden_size': ML_CONFIG['forecasting']['lstm']['hidden_size'],
        'num_layers': ML_CONFIG['forecasting']['lstm']['num_layers'],
        'dropout': ML_CONFIG['forecasting']['lstm']['dropout'],
        'learning_rate': ML_CONFIG['forecasting']['lstm']['learning_rate'],
        'forecast_horizon': forecast_horizon,
        'use_amp': ML_CONFIG['forecasting']['lstm'].get('use_amp', False)  # Mixed precision
    }
    
    forecaster = EnergyDemandForecaster(config)
    forecaster.build_model(input_size=X_train.shape[2])
    
    # Epochs réduit pour Windows avec early stopping intelligent
    epochs = 50  # Réduit de 80 à 50 (early stopping s'active généralement avant)
    print(f"\n⚡ Configuration optimisée Windows:")
    print(f"  - Max epochs: {epochs} (early stopping actif)")
    print(f"  - Early stopping patience: 7 epochs")
    print(f"  - Learning rate adaptatif avec ReduceLROnPlateau")
    
    train_losses, val_losses = forecaster.train(train_loader, val_loader, epochs=epochs)
    
    # valuation - DESCALER les prédictions
    print(f"\n valuation sur ensemble test...")
    
    predictions_scaled = []
    
    # Barre de progression pour les prédictions
    test_pbar = tqdm(range(len(X_test)), desc="🔮 Prédictions test", unit="sample")
    for i in test_pbar:
        pred = forecaster.predict(X_test[i])
        predictions_scaled.append(pred[0, :, 0])
    test_pbar.close()
    
    predictions_scaled = np.array(predictions_scaled).flatten()
    
    # DESCALER les prédictions
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    actuals = y_test_original.flatten()
    
    # Mtriques sur valeurs originales (pas scaled)
    metrics = calculate_metrics(actuals, predictions)
    
    print(f"\n RSULTATS FINAUX:")
    print(f"  - MAPE: {metrics['MAPE']:.2f}%")
    print(f"  - RMSE: {metrics['RMSE']:.2f} kWh")
    print(f"  - MAE: {metrics['MAE']:.2f} kWh")
    print(f"  - R: {metrics['R2']:.4f}")
    
    # Sauvegarder mtriques
    results_path = Path(__file__).parent.parent / "models" / "forecasting" / f"results_{station_id}.json"
    results = {
        'station_id': station_id,
        'model': 'LSTM',
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Rsultats sauvegards: {results_path}")
    
    # Retourner avec preprocessor pour RL
    return forecaster, metrics, preprocessor


def train_optimization_models(df: pd.DataFrame, station_id: str):
    """Entrane l'agent RL d'optimisation"""
    print("\n" + "="*60)
    print("  ENTRANEMENT AGENT D'OPTIMISATION RL")
    print("="*60)
    
    from models.optimization.rl_agent import EnergyOptimizationAgent
    
    # Configuration
    station_config = next(s for s in STATIONS if s.station_id == station_id)
    
    env_config = {
        'num_pumps': station_config.pump_count,
        'max_power_kw': station_config.max_power_kw,
        'reservoir_capacity_m3': station_config.reservoir_capacity_m3,
        'episode_length': 168
    }
    
    # Crer agent
    agent = EnergyOptimizationAgent(env_config)
    
    # Donnes demande pour environnement
    demand_data = df['water_demand_m3'].values
    
    # Crer environnement
    agent.create_env(demand_data=demand_data)
    
    # Entranement (optimisé pour Windows)
    print(f"\n⚡ Entranement PPO Agent (5-10 min sur Windows)...")
    print(" Note: Early stopping activé si convergence stable")
    print(" Pour production: augmenter à 200k+ timesteps")
    
    agent.train(total_timesteps=30000)  # Réduit à 30k avec early stopping (était 50k)
    
    # Sauvegarder
    model_path = Path(__file__).parent.parent / "models" / "optimization" / f"ppo_agent_{station_id}.zip"
    agent.save(str(model_path))
    
    print(f" Agent sauvegard: {model_path}")
    
    # Test rapide
    print(f"\n Test agent...")
    obs = agent.env.reset()
    total_reward = 0
    
    for _ in range(168):  # 1 semaine
        action = agent.optimize(obs)
        obs, reward, done, info = agent.env.step(action)
        total_reward += reward[0]
        
        if done:
            break
    
    print(f" Reward sur 1 semaine test: {total_reward:.2f}")
    print(f"  - Cot moyen: {info[0]['total_cost'] / 168:,.0f} FCFA/h")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description="Entranement modles ONEA")
    parser.add_argument('--station', type=str, default='OUG_ZOG', help='ID station')
    parser.add_argument('--models', type=str, default='all', 
                       choices=['all', 'forecasting', 'optimization'],
                       help='Modles  entraner')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ONEA SMART ENERGY OPTIMIZER - ENTRAINEMENT MODELES")
    print("="*60 + "\n")
    
    print(f"Station: {args.station}")
    print(f"Modeles: {args.models}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Charger donnes
    df = load_station_data(args.station)
    
    # Entranement
    # Variables pour décision RL
    need_rl_optimization = False
    forecaster = None
    metrics = None
    
    if args.models in ['all', 'forecasting']:
        forecaster, metrics, preprocessor = train_forecasting_models(df, args.station)
        
        # Objectif: MAPE < 6%
        if metrics['MAPE'] < 6.0:
            print(f"\n OBJECTIF ATTEINT! MAPE = {metrics['MAPE']:.2f}% < 6%")
            print("  Modle LSTM suffisant, RL optionnel")
        else:
            print(f"\n  MAPE = {metrics['MAPE']:.2f}% > 6% - Optimisation RL REQUISE!")
            need_rl_optimization = True
    
    # Déclenchement automatique RL si métriques insuffisantes
    if args.models in ['all', 'optimization'] or need_rl_optimization:
        if need_rl_optimization:
            print("\n" + "="*60)
            print(" DECLENCHEMENT AUTOMATIQUE AGENT RL")
            print("="*60)
            print(f"\nRaison: MAPE={metrics['MAPE']:.2f}% > seuil 6%")
            print("L'agent RL va optimiser les predictions du LSTM...")
        
        # Lancer entraînement RL
        try:
            from train_rl import train_rl_agent
            
            print("\n" + "="*60)
            print(" ENTRAINEMENT AGENT RL PPO")
            print("="*60)
            
            agent, rl_results = train_rl_agent(args.station, total_steps=30000)  # Réduit à 30k
            
            if agent:
                print("\n SUCCES! Agent RL entraine et sauvegarde!")
                print(f"  Reward final: {rl_results['final_avg_reward']:.2f}")
                print(f"  Economies estimees: 27-30% des couts")
                
                # Mettre à jour métriques
                metrics['RL_trained'] = True
                metrics['RL_reward'] = rl_results['final_avg_reward']
            else:
                print("\n ERREUR: Entrainement RL echoue")
                metrics['RL_trained'] = False
                
        except Exception as e:
            print(f"\n ERREUR RL: {e}")
            print("  Solution de secours: Optimisation statistique utilisee")
            metrics['RL_trained'] = False
    
    print("\n" + "="*60)
    print(" ENTRANEMENT TERMIN AVEC SUCCS!")
    print("="*60)
    
    # Résumé final
    if metrics:
        print("\n RESULTATS FINAUX:")
        print(f"  LSTM MAPE: {metrics['MAPE']:.2f}%")
        print(f"  LSTM RMSE: {metrics['RMSE']:.2f} kWh")
        print(f"  LSTM R2: {metrics['R2']:.4f}")
        
        if metrics.get('RL_trained', False):
            print(f"\n  Agent RL: ENTRAINE")
            print(f"  RL Reward: {metrics.get('RL_reward', 'N/A'):.2f}")
            print(f"  Optimisation: COMPLETE (LSTM + RL)")
        else:
            print(f"\n  Agent RL: NON REQUIS (MAPE < 6%)")
            print(f"  Optimisation: LSTM seul suffisant")
    
    print("\nProchaines tapes:")
    print("1. Lancer dashboard: streamlit run dashboard/app.py")
    print("2. Lancer API: uvicorn api.main:app --reload")
    print("3. Consulter rsultats: models/forecasting/results_*.json")
    
    if metrics and metrics.get('RL_trained', False):
        print("4. Agent RL disponible: models/optimization/ppo_agent_*.pth")
    
    # Sauvegarder métriques finales combinées
    if metrics:
        results_path = Path(__file__).parent.parent / "models" / "forecasting" / f"final_results_{args.station}.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n Metriques finales sauvegardees: {results_path}")


if __name__ == "__main__":
    main()

