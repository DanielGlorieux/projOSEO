"""
Agent de Reinforcement Learning pour optimisation énergétique
Utilise PPO (Proximal Policy Optimization) - état de l'art RL
"""
# Patch pour éviter tensorboard
import os
os.environ['TENSORBOARD_DISABLE'] = '1'

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import avec gestion d'erreur tensorboard
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
except AttributeError as e:
    if 'tensorboard' in str(e).lower() or 'tensorflow' in str(e).lower():
        # Désactiver tensorboard dans torch
        import sys
        import torch.utils.tensorboard
        # Mock tensorboard
        class MockSummaryWriter:
            def __init__(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        torch.utils.tensorboard.SummaryWriter = MockSummaryWriter
        
        # Réimporter
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
    else:
        raise

from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.config import ENERGY_PRICING, get_energy_price


class PumpOptimizationEnv(gym.Env):
    """
    Environnement Gym pour optimisation du pompage
    
    State: [heure, jour_semaine, demande_prévue, niveau_réservoir, 
            prix_énergie, pompes_actives, efficacité_moyenne]
    
    Action: [nombre_pompes_à_activer (0 à n_pomps), mode_puissance (0-1)]
    
    Reward: -coût_énergétique -pénalité_service
    """
    
    def __init__(self, 
                 num_pumps: int = 4,
                 max_power_kw: float = 850.0,
                 reservoir_capacity_m3: float = 5000.0,
                 demand_data: np.ndarray = None,
                 episode_length: int = 168):  # 1 semaine
        
        super(PumpOptimizationEnv, self).__init__()
        
        self.num_pumps = num_pumps
        self.max_power_kw = max_power_kw
        self.reservoir_capacity = reservoir_capacity_m3
        self.demand_data = demand_data if demand_data is not None else self._generate_dummy_demand()
        self.episode_length = episode_length
        
        # State space: 7 features
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0.5]),
            high=np.array([23, 6, reservoir_capacity_m3 * 2, reservoir_capacity_m3, 150, num_pumps, 1.0]),
            dtype=np.float32
        )
        
        # Action space: nombre de pompes (discret) + mode puissance (continu)
        self.action_space = spaces.Box(
            low=np.array([0, 0.3]),  # Min 1 pompe, 30% puissance
            high=np.array([num_pumps, 1.0]),  # Max all pompes, 100% puissance
            dtype=np.float32
        )
        
        self.reset()
        
    def _generate_dummy_demand(self) -> np.ndarray:
        """Génère demande synthétique pour test"""
        t = np.arange(1000)
        demand = 2000 + 500 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 100, 1000)
        return np.maximum(demand, 0)
    
    def reset(self, seed=None, options=None):
        """Reset environnement"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.reservoir_level = self.reservoir_capacity * 0.7  # Commencer à 70%
        self.total_cost = 0.0
        self.total_energy = 0.0
        self.penalties = 0
        
        # Position aléatoire dans données
        self.data_offset = np.random.randint(0, len(self.demand_data) - self.episode_length)
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Retourne l'état actuel"""
        current_time_idx = self.data_offset + self.current_step
        
        hour = current_time_idx % 24
        day_of_week = (current_time_idx // 24) % 7
        current_demand = self.demand_data[current_time_idx]
        energy_price = get_energy_price(hour)
        
        # État simplifié
        obs = np.array([
            hour,
            day_of_week,
            current_demand,
            self.reservoir_level,
            energy_price,
            self.num_pumps / 2,  # État initial: ~50% pompes
            0.85  # Efficacité moyenne initiale
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        """Exécute une action"""
        # Décoder action
        num_pumps_active = int(np.clip(action[0], 1, self.num_pumps))
        power_mode = float(np.clip(action[1], 0.3, 1.0))
        
        # État actuel
        current_time_idx = self.data_offset + self.current_step
        hour = current_time_idx % 24
        current_demand = self.demand_data[current_time_idx]
        energy_price = get_energy_price(hour)
        
        # Production eau (basée sur pompes actives)
        pump_power_per_unit = self.max_power_kw / self.num_pumps
        total_power_kw = num_pumps_active * pump_power_per_unit * power_mode
        
        # Efficacité (dégradée si trop/trop peu de pompes)
        optimal_pumps = max(1, int(current_demand / (self.reservoir_capacity / self.num_pumps)))
        efficiency_penalty = abs(num_pumps_active - optimal_pumps) * 0.05
        efficiency = 0.85 - efficiency_penalty
        efficiency = np.clip(efficiency, 0.60, 0.90)
        
        # Production eau (kWh -> m³, ~0.4 kWh/m³)
        water_produced = (total_power_kw * efficiency) / 0.4
        
        # Mise à jour réservoir
        self.reservoir_level += water_produced - current_demand
        self.reservoir_level = np.clip(self.reservoir_level, 0, self.reservoir_capacity)
        
        # Coût énergétique
        energy_kwh = total_power_kw  # Sur 1h
        power_factor = 0.82 + (efficiency - 0.75) * 0.2  # Corrélé à efficacité
        
        # Coût de base
        cost = energy_kwh * energy_price
        
        # Pénalité facteur de puissance
        if power_factor < ENERGY_PRICING.power_factor_penalty_threshold:
            cost *= 1.15
            self.penalties += 1
        
        self.total_cost += cost
        self.total_energy += energy_kwh
        
        # REWARD FUNCTION (key for performance)
        reward = 0.0
        
        # 1. Minimiser coût (objectif principal)
        reward -= cost / 100  # Normalisation
        
        # 2. Pénalité niveau réservoir critique
        if self.reservoir_level < self.reservoir_capacity * 0.2:
            reward -= 50  # Critique: service en danger
        elif self.reservoir_level < self.reservoir_capacity * 0.3:
            reward -= 10  # Avertissement
        
        # 3. Bonus maintien niveau optimal (50-80%)
        if 0.5 <= (self.reservoir_level / self.reservoir_capacity) <= 0.8:
            reward += 2
        
        # 4. Bonus efficacité
        if efficiency > 0.82:
            reward += 1
        
        # 5. Pénalité surutilisation heures pleines
        if hour in ENERGY_PRICING.peak_hours and num_pumps_active > optimal_pumps:
            reward -= 5
        
        # 6. Bonus utilisation heures creuses
        if hour in ENERGY_PRICING.off_peak_hours:
            reward += 1
        
        # Termination
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Pénalité finale si service défaillant
        if terminated and self.reservoir_level < self.reservoir_capacity * 0.25:
            reward -= 100
        
        info = {
            'total_cost': self.total_cost,
            'total_energy': self.total_energy,
            'penalties': self.penalties,
            'avg_reservoir_level': self.reservoir_level / self.reservoir_capacity,
            'avg_efficiency': efficiency
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        """Affichage (optionnel)"""
        pass


class EnergyOptimizationAgent:
    """
    Agent RL pour optimisation énergétique
    """
    
    def __init__(self, env_config: Dict):
        self.env_config = env_config
        self.env = None
        self.model = None
        
    def create_env(self, demand_data: np.ndarray = None):
        """Crée l'environnement"""
        env = PumpOptimizationEnv(
            num_pumps=self.env_config.get('num_pumps', 4),
            max_power_kw=self.env_config.get('max_power_kw', 850.0),
            reservoir_capacity_m3=self.env_config.get('reservoir_capacity_m3', 5000.0),
            demand_data=demand_data,
            episode_length=self.env_config.get('episode_length', 168)
        )
        
        self.env = DummyVecEnv([lambda: env])
        print(f"✓ Environnement créé")
        print(f"  - Observation space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.shape}")
        
        return self.env
    
    def train(self, total_timesteps: int = 100000):
        """Entraîne l'agent PPO"""
        if self.env is None:
            raise ValueError("Créer l'environnement d'abord avec create_env()")
        
        # Créer agent PPO (sans tensorboard)
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=None  # Désactivé pour éviter conflits
        )
        
        print(f"\n🚀 Entraînement PPO - {total_timesteps:,} timesteps")
        print("⚠️  Tensorboard désactivé (conflits dépendances)")

        eval_env = self.create_env()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/optimization/',
            log_path='./logs/',
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Entraînement
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        print("✓ Entraînement terminé")
        
        return self.model
    
    def optimize(self, state: np.ndarray) -> np.ndarray:
        """Recommande action optimale"""
        if self.model is None:
            raise ValueError("Entraîner ou charger modèle d'abord")
        
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def save(self, filepath: str):
        """Sauvegarde le modèle"""
        if self.model:
            self.model.save(filepath)
            print(f"✓ Modèle sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charge le modèle"""
        self.model = PPO.load(filepath, env=self.env)
        print(f"✓ Modèle chargé: {filepath}")


if __name__ == "__main__":
    print("🧪 Test Agent d'Optimisation RL\n")
    
    # Configuration
    env_config = {
        'num_pumps': 4,
        'max_power_kw': 850.0,
        'reservoir_capacity_m3': 5000.0,
        'episode_length': 168
    }
    
    # Créer agent
    agent = EnergyOptimizationAgent(env_config)
    
    # Test environnement
    env = agent.create_env()
    
    # Test épisode aléatoire
    obs = env.reset()
    total_reward = 0
    
    print("\n🎮 Simulation 24h avec actions aléatoires:")
    for step in range(24):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if (step + 1) % 6 == 0:
            print(f"  Step {step+1}: Reward={reward[0]:.2f}, Total={total_reward:.2f}")
    
    print(f"\n✓ Test réussi! Reward total: {total_reward:.2f}")
    print("\nPour entraîner: agent.train(total_timesteps=100000)")
