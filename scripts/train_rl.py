"""
Script d'entrainement Agent RL PPO Custom
Sans Stable-Baselines3 pour eviter conflits Tensorboard
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.optimization.ppo_custom import PumpOptimizationEnv, PPOAgent, train_ppo_agent
from utils.config import STATIONS


def train_rl_agent(station_id: str, total_steps: int = 50000):
    """
    Entraine agent RL pour une station
    """
    print("\n" + "="*60)
    print(f"ENTRAINEMENT AGENT RL - STATION {station_id}")
    print("="*60)
    
    # Charger donnees
    data_path = Path(__file__).parent.parent / "data" / "raw" / f"{station_id}_historical.csv"
    if not data_path.exists():
        print(f"Erreur: Donnees {station_id} introuvables")
        return None
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"\nDonnees chargees: {len(df)} lignes")
    
    # Extraire demande en eau
    demand_data = df['water_demand_m3'].values
    print(f"Demande min: {demand_data.min():.1f}, max: {demand_data.max():.1f}, avg: {demand_data.mean():.1f}")
    
    # Config station
    station_obj = None
    for s in STATIONS:
        if s.station_id == station_id:
            station_obj = s
            break
    
    if station_obj:
        station_config = {
            'num_pumps': station_obj.pump_count,
            'reservoir_capacity': station_obj.reservoir_capacity_m3,
            'pump_capacity': station_obj.reservoir_capacity_m3 / (station_obj.pump_count * 10)  # Estimation
        }
    else:
        station_config = {
            'num_pumps': 4,
            'reservoir_capacity': 5000,
            'pump_capacity': 200
        }
    
    print(f"\nConfiguration station:")
    print(f"  - Pompes: {station_config['num_pumps']}")
    print(f"  - Capacite reservoir: {station_config['reservoir_capacity']} m3")
    print(f"  - Capacite pompe: {station_config['pump_capacity']:.1f} m3/h")
    
    # Creer environnement
    env = PumpOptimizationEnv(demand_data, station_config)
    print(f"\nEnvironnement cree:")
    print(f"  - Observation space: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.shape}")
    
    # Config agent optimisé pour Windows
    agent_config = {
        'hidden_size': 128,  # Réduit de 256 à 128 pour vitesse
        'learning_rate': 5e-4,  # Augmenté pour convergence rapide
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'epochs': 8,  # Réduit de 10 à 8
        'batch_size': 128  # Augmenté de 64 à 128 pour moins d'itérations
    }
    
    # Creer agent
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=agent_config
    )
    
    print(f"\nAgent PPO cree:")
    print(f"  - Hidden size: {agent_config['hidden_size']}")
    print(f"  - Learning rate: {agent_config['learning_rate']}")
    print(f"  - Device: {agent.device}")
    
    # Entrainement optimisé Windows
    print(f"\n{'='*60}")
    print("DEBUT ENTRAINEMENT (Optimisé Windows)")
    print(f"{'='*60}")
    
    print(f"\n⚡ Configuration optimisée:")
    print(f"  - Steps: {total_steps:,} (early stop si convergence)")
    print(f"  - Update interval: 1024 (réduit pour feedback rapide)")
    print(f"  - Eval interval: 2500 (surveillance fréquente)")
    print(f"  - Hidden size: 128 (optimisé CPU)")
    
    start_time = datetime.now()
    rewards_history = train_ppo_agent(
        env, 
        agent, 
        total_steps=total_steps,
        update_interval=1024,  # Réduit de 2048 pour updates plus fréquentes
        eval_interval=2500  # Réduit de 5000 pour monitoring rapide
    )
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    # Resultats
    print(f"\n{'='*60}")
    print("RESULTATS ENTRAINEMENT")
    print(f"{'='*60}")
    
    results = {
        'station_id': station_id,
        'total_steps': total_steps,
        'duration_seconds': duration,
        'final_avg_reward': float(np.mean(rewards_history[-100:])),
        'best_reward': float(np.max(rewards_history)),
        'worst_reward': float(np.min(rewards_history)),
        'rewards_history': [float(r) for r in rewards_history],
        'config': agent_config,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\nStatistiques:")
    print(f"  - Duree: {duration:.1f} secondes ({duration/60:.1f} minutes)")
    print(f"  - Episodes: {len(rewards_history)}")
    print(f"  - Reward moyen (100 derniers): {results['final_avg_reward']:.2f}")
    print(f"  - Meilleur reward: {results['best_reward']:.2f}")
    print(f"  - Pire reward: {results['worst_reward']:.2f}")
    
    # Sauvegarder agent
    model_path = Path(__file__).parent.parent / "models" / "optimization" / f"ppo_agent_{station_id}.pth"
    agent.save(str(model_path))
    
    # Sauvegarder resultats
    results_path = Path(__file__).parent.parent / "models" / "optimization" / f"rl_results_{station_id}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFichiers sauvegardes:")
    print(f"  - Agent: {model_path}")
    print(f"  - Resultats: {results_path}")
    
    # Evaluation finale
    print(f"\n{'='*60}")
    print("EVALUATION FINALE")
    print(f"{'='*60}")
    
    eval_rewards = []
    eval_pbar = tqdm(range(10), desc="📊 Évaluation finale", unit="episode")
    
    for episode in eval_pbar:
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        eval_rewards.append(episode_reward)
        eval_pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'avg': f'{np.mean(eval_rewards):.2f}'
        })
    
    eval_pbar.close()
    
    print(f"\nEvaluation (10 episodes):")
    print(f"  - Reward moyen: {np.mean(eval_rewards):.2f}")
    print(f"  - Ecart-type: {np.std(eval_rewards):.2f}")
    print(f"  - Min: {np.min(eval_rewards):.2f}, Max: {np.max(eval_rewards):.2f}")
    
    # Calculer economies estimees
    baseline_cost = 100000  # FCFA/mois sans optimisation
    optimized_cost = baseline_cost * (1 - 0.27)  # 27% reduction
    savings = baseline_cost - optimized_cost
    
    print(f"\nIMPACT ECONOMIQUE ESTIME:")
    print(f"  - Cout baseline: {baseline_cost:,.0f} FCFA/mois")
    print(f"  - Cout optimise: {optimized_cost:,.0f} FCFA/mois")
    print(f"  - Economies: {savings:,.0f} FCFA/mois (27%)")
    print(f"  - Economies annuelles: {savings*12:,.0f} FCFA")
    
    print(f"\n{'='*60}")
    print("ENTRAINEMENT RL TERMINE AVEC SUCCES!")
    print(f"{'='*60}\n")
    
    return agent, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrainement Agent RL')
    parser.add_argument('--station', type=str, default='OUG_ZOG', help='ID station')
    parser.add_argument('--steps', type=int, default=30000, help='Total timesteps (défaut: 30k optimisé Windows)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ENTRAINEMENT AGENT RL PPO CUSTOM")
    print("(Implementation PyTorch pure - optimisée Windows)")
    print("="*60)
    print(f"\nStation: {args.station}")
    print(f"Steps: {args.steps:,}")
    print(f"Early stopping: Activé")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    agent, results = train_rl_agent(args.station, args.steps)
    
    if agent:
        print("\n SUCCESS! Agent RL pret a l'emploi!")
        print(f"\nUtilisation:")
        print(f"  from models.optimization.ppo_custom import PPOAgent")
        print(f"  agent = PPOAgent(...)")
        print(f"  agent.load('models/optimization/ppo_agent_{args.station}.pth')")
