"""
Agent RL PPO Custom (sans Stable-Baselines3)
Implementation pure PyTorch pour eviter conflits Tensorboard
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import sys
import pickle
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.config import ENERGY_PRICING, get_energy_price


class PumpOptimizationEnv(gym.Env):
    """
    Environnement Gymnasium pour optimisation pompes
    """
    
    def __init__(self, demand_data: np.ndarray, station_config: dict):
        super().__init__()
        
        self.demand_data = demand_data
        self.num_pumps = station_config.get('num_pumps', 4)
        self.reservoir_capacity = station_config.get('reservoir_capacity', 5000)
        self.pump_capacity = station_config.get('pump_capacity', 200)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.3]), 
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = len(demand_data) - 1
        self.reservoir_level = self.reservoir_capacity * 0.5
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.reservoir_level = self.reservoir_capacity * 0.5
        return self._get_obs(), {}
    
    def _get_obs(self):
        idx = min(self.current_step, self.max_steps - 1)
        hour = idx % 24
        day_of_week = (idx // 24) % 7
        demand = self.demand_data[idx]
        price = get_energy_price(hour)
        
        obs = np.array([
            hour / 24.0,
            day_of_week / 7.0,
            demand / 3000.0,
            self.reservoir_level / self.reservoir_capacity,
            price / 100.0,
            0.5,
            0.85
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: np.ndarray):
        # Action: [num_pumps_ratio, power_ratio]
        num_pumps = int(np.clip(action[0] * self.num_pumps, 1, self.num_pumps))
        power = np.clip(action[1], 0.3, 1.0)
        
        # Calculs
        idx = min(self.current_step, self.max_steps - 1)
        demand = self.demand_data[idx]
        hour = idx % 24
        price = get_energy_price(hour)
        
        production = num_pumps * self.pump_capacity * power
        consumption_kwh = num_pumps * 50 * power
        cost = consumption_kwh * price
        
        # Reservoir
        self.reservoir_level += (production - demand)
        self.reservoir_level = np.clip(self.reservoir_level, 0, self.reservoir_capacity)
        
        # REWARD NORMALISÉ + SHAPING
        # 1. Coût énergétique (normalisé)
        cost_penalty = -cost / 100.0  # Divisé par 100 au lieu de 1000
        
        # 2. Pénalité shortage (critique)
        shortage = max(0, demand - production)
        shortage_penalty = -shortage * 0.5  # Moins agressif
        
        # 3. Réservoir optimal (bonus/malus)
        tank_level = self.reservoir_level / self.reservoir_capacity
        if 0.4 <= tank_level <= 0.8:
            reservoir_reward = 5.0  # BONUS si optimal
        else:
            reservoir_reward = -abs(tank_level - 0.6) * 10.0  # Pénalité si hors optimal
        
        # 4. Bonus efficacité (éviter gaspillage)
        overproduction = max(0, production - demand)
        if overproduction < demand * 0.1:  # <10% surproduction
            efficiency_bonus = 3.0
        else:
            efficiency_bonus = -overproduction * 0.1
        
        # 5. Bonus off-peak (encourager utilisation heures creuses)
        if hour in [0, 1, 2, 3, 4, 5, 23]:
            offpeak_bonus = 2.0
        else:
            offpeak_bonus = 0.0
        
        reward = (cost_penalty + shortage_penalty + reservoir_reward + 
                 efficiency_bonus + offpeak_bonus)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}


class ActorCritic(nn.Module):
    """
    Reseau Actor-Critic pour PPO
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Sigmoid()  # Actions entre 0 et 1
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, obs):
        shared_features = self.shared(obs)
        return shared_features
    
    def act(self, obs):
        shared_features = self.forward(obs)
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_logstd)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_logprob
    
    def evaluate(self, obs, action):
        shared_features = self.forward(obs)
        
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_value = self.critic(shared_features).squeeze()
        
        return action_logprobs, state_value, dist_entropy


class PPOAgent:
    """
    Agent PPO custom (sans Stable-Baselines3)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(obs_dim, action_dim, config['hidden_size']).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['learning_rate'])
        
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_epsilon = config['clip_epsilon']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        
        self.buffer = {
            'obs': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
    def select_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, logprob = self.policy.act(obs_tensor)
            value = self.policy.critic(self.policy.forward(obs_tensor)).squeeze()
        
        return action.cpu().numpy()[0], logprob.cpu().numpy(), value.cpu().numpy()
    
    def store_transition(self, obs, action, logprob, reward, done, value):
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(logprob)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['values'].append(value)
    
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self):
        if len(self.buffer['obs']) == 0:
            return {'loss': 0}
        
        # Convertir buffer en tensors
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.buffer['logprobs'])).to(self.device)
        
        # GAE
        advantages, returns = self.compute_gae(
            self.buffer['rewards'],
            self.buffer['values'],
            self.buffer['dones']
        )
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normaliser advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(self.epochs):
            # Evaluate
            logprobs, values, entropy = self.policy.evaluate(obs, actions)
            
            # Ratio
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Total loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((returns - values) ** 2).mean()
            entropy_loss = -0.01 * entropy.mean()
            
            loss = actor_loss + critic_loss + entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear buffer
        for key in self.buffer:
            self.buffer[key] = []
        
        return {
            'loss': total_loss / self.epochs,
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Agent sauvegarde: {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Agent charge: {path}")


def train_ppo_agent(env: gym.Env, agent: PPOAgent, total_steps: int = 200000, 
                   update_interval: int = 2048, eval_interval: int = 10000):
    """
    Entrainement agent PPO - Optimisé Windows avec early stopping
    """
    print(f"\nEntrainement PPO Agent...")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Update interval: {update_interval}")
    print(f"  - Device: {agent.device}")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    step_count = 0
    
    rewards_history = []
    best_reward = -float('inf')
    
    # Early stopping pour RL
    convergence_patience = 5
    convergence_counter = 0
    min_episodes = 20  # Minimum d'épisodes avant early stop
    
    # Barre de progression pour les steps
    pbar = tqdm(total=total_steps, desc="🤖 Entraînement RL PPO", unit="step")
    
    while step_count < total_steps:
        # Select action
        action, logprob, value = agent.select_action(obs)
        
        # Step
        next_obs, reward, done, truncated, _ = env.step(action)
        
        # Store
        agent.store_transition(obs, action, logprob, reward, done or truncated, value)
        
        episode_reward += reward
        step_count += 1
        pbar.update(1)
        
        # Update policy
        if step_count % update_interval == 0:
            metrics = agent.update()
            if step_count % eval_interval == 0:
                avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history) if rewards_history else 0
                
                # Mise à jour barre avec métriques
                pbar.set_postfix({
                    'episodes': episode_count,
                    'avg_reward': f'{avg_reward:.2f}',
                    'best': f'{best_reward:.2f}',
                    'loss': f'{metrics["loss"]:.4f}'
                })
                
                # Track best et early stopping
                if avg_reward > best_reward:
                    improvement = avg_reward - best_reward
                    best_reward = avg_reward
                    convergence_counter = 0
                    pbar.write(f"🎯 Nouveau meilleur reward: {best_reward:.2f} (step {step_count})")
                else:
                    convergence_counter += 1
                
                # Early stopping si convergence stable
                if episode_count >= min_episodes and convergence_counter >= convergence_patience:
                    pbar.write(f"✋ Early stopping: convergence stable (reward={avg_reward:.2f})")
                    break
                
                # Arrêt si performance excellente
                if avg_reward > 50 and episode_count >= min_episodes:
                    pbar.write(f"🏆 Performance excellente atteinte! (reward={avg_reward:.2f})")
                    break
        
        if done or truncated:
            rewards_history.append(episode_reward)
            episode_count += 1
            episode_reward = 0
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    pbar.close()
    
    print(f"\n✅ Entrainement termine!")
    print(f"  - Episodes: {episode_count}")
    print(f"  - Steps effectués: {step_count:,}/{total_steps:,}")
    print(f"  - Avg Reward (last 100): {np.mean(rewards_history[-100:]):.2f}")
    print(f"  - Best Avg Reward: {best_reward:.2f}")
    
    return rewards_history


if __name__ == "__main__":
    print("Test Agent PPO Custom...")
    
    # Donnees test
    demand_data = np.random.uniform(1000, 2500, 1000)
    
    station_config = {
        'num_pumps': 4,
        'reservoir_capacity': 5000,
        'pump_capacity': 200
    }
    
    # Environnement
    env = PumpOptimizationEnv(demand_data, station_config)
    
    # Agent
    config = {
        'hidden_size': 256,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'epochs': 10,
        'batch_size': 64
    }
    
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        config=config
    )
    
    # Entrainement court
    print("\nTest entrainement (5000 steps)...")
    rewards = train_ppo_agent(env, agent, total_steps=5000, update_interval=512)
    
    print(f"\nTest reussi! Rewards finaux: {np.mean(rewards[-10:]):.2f}")
