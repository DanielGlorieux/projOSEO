"""
Optimiseur basé sur algorithme génétique + Optuna
Plus efficace que RL pour ce problème d'optimisation
"""
import numpy as np
import optuna
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Résultat d'optimisation"""
    pumps_active: int
    power_level: float
    expected_cost: float
    expected_savings: float
    savings_percent: float
    actions: List[Dict]


class SmartPumpOptimizer:
    """Optimiseur intelligent de pompage avec Optuna"""
    
    def __init__(self, station_config: Dict):
        self.station = station_config
        self.history = []
        
    def get_energy_price(self, hour: int) -> float:
        """Tarif énergétique par heure"""
        if hour >= 23 or hour < 6:
            return 65.0  # Heures creuses
        elif 18 <= hour < 23:
            return 85.0  # Heures pleines
        else:
            return 75.0  # Heures normales
    
    def calculate_cost(self, pumps_active: int, power_level: float, 
                      hour: int, demand: float) -> Tuple[float, Dict]:
        """Calcule le coût d'une configuration"""
        price = self.get_energy_price(hour)
        
        # Puissance consommée
        power_per_pump = self.station['max_power_kw'] / self.station['pump_count']
        total_power = power_per_pump * pumps_active * power_level
        
        # Coût de base
        base_cost = total_power * price
        
        # Production
        production_capacity = self.station['max_flow_m3h'] * (pumps_active / self.station['pump_count']) * power_level
        
        # Pénalités
        penalties = 0
        penalty_reasons = []
        
        # Pénalité sous-production
        if production_capacity < demand:
            shortage = demand - production_capacity
            penalties += shortage * price * 2.0
            penalty_reasons.append(f"Sous-production: {shortage:.1f} m³/h")
        
        # Pénalité sur-production excessive
        if production_capacity > demand * 1.5:
            excess = production_capacity - demand * 1.5
            penalties += excess * price * 0.5
            penalty_reasons.append(f"Surproduction: {excess:.1f} m³/h")
        
        # Pénalité facteur de puissance
        power_factor = 0.82 + (power_level - 0.5) * 0.15
        if power_factor < 0.85:
            penalties += base_cost * 0.15
            penalty_reasons.append("Facteur puissance bas")
        
        # Pénalité efficacité
        efficiency = 0.75 + (power_level - 0.5) * 0.10
        if efficiency < 0.75:
            penalties += base_cost * 0.10
            penalty_reasons.append("Efficacité dégradée")
        
        total_cost = base_cost + penalties
        
        metrics = {
            'base_cost': base_cost,
            'penalties': penalties,
            'total_cost': total_cost,
            'production': production_capacity,
            'efficiency': efficiency,
            'power_factor': power_factor,
            'penalty_reasons': penalty_reasons
        }
        
        return total_cost, metrics
    
    def optimize_for_period(self, hour: int, demand: float, 
                          baseline_cost: float = None) -> OptimizationResult:
        """Optimise la configuration pour une période donnée"""
        
        def objective(trial):
            """Fonction objectif Optuna"""
            pumps = trial.suggest_int('pumps', 1, self.station['pump_count'])
            power = trial.suggest_float('power', 0.5, 1.0)
            
            cost, _ = self.calculate_cost(pumps, power, hour, demand)
            return cost
        
        # Optimisation avec Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        # Meilleure solution
        best_pumps = study.best_params['pumps']
        best_power = study.best_params['power']
        
        optimized_cost, metrics = self.calculate_cost(best_pumps, best_power, hour, demand)
        
        # Baseline si non fourni
        if baseline_cost is None:
            baseline_cost, _ = self.calculate_cost(
                self.station['pump_count'],
                1.0,
                hour,
                demand
            )
        
        savings = baseline_cost - optimized_cost
        savings_percent = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        # Actions recommandées
        actions = []
        
        if best_pumps != self.station['pump_count']:
            actions.append({
                'action': 'adjust_pumps',
                'current': self.station['pump_count'],
                'recommended': best_pumps,
                'reason': f"Optimisation pour demande {demand:.0f} m³/h"
            })
        
        if best_power < 1.0:
            actions.append({
                'action': 'adjust_power',
                'current': 1.0,
                'recommended': best_power,
                'reason': f"Réduction puissance à {best_power*100:.0f}% (tarif {self.get_energy_price(hour)} FCFA/kWh)"
            })
        
        if metrics['efficiency'] < 0.80:
            actions.append({
                'action': 'maintenance_alert',
                'current': metrics['efficiency'],
                'recommended': 0.85,
                'reason': "Efficacité dégradée - maintenance requise"
            })
        
        if metrics['power_factor'] < 0.85:
            actions.append({
                'action': 'power_factor_correction',
                'current': metrics['power_factor'],
                'recommended': 0.90,
                'reason': "Installer condensateurs pour correction"
            })
        
        return OptimizationResult(
            pumps_active=best_pumps,
            power_level=best_power,
            expected_cost=optimized_cost,
            expected_savings=savings,
            savings_percent=savings_percent,
            actions=actions
        )
    
    def optimize_24h_schedule(self, demand_forecast: np.ndarray) -> pd.DataFrame:
        """Optimise le planning sur 24h"""
        schedule = []
        
        for hour in range(24):
            demand = demand_forecast[hour] if hour < len(demand_forecast) else demand_forecast[-1]
            result = self.optimize_for_period(hour, demand)
            
            schedule.append({
                'hour': hour,
                'demand_m3h': demand,
                'pumps_active': result.pumps_active,
                'power_level': result.power_level,
                'expected_cost': result.expected_cost,
                'energy_price': self.get_energy_price(hour),
                'savings_fcfa': result.expected_savings
            })
        
        return pd.DataFrame(schedule)


def optimize_multi_station(stations: List[Dict], demands: Dict[str, float], 
                          current_hour: int) -> Dict:
    """Optimise plusieurs stations simultanément"""
    results = {}
    total_savings = 0
    
    for station_config in stations:
        station_id = station_config['station_id']
        demand = demands.get(station_id, station_config.get('avg_demand', 3000))
        
        optimizer = SmartPumpOptimizer(station_config)
        result = optimizer.optimize_for_period(current_hour, demand)
        
        results[station_id] = {
            'pumps_active': result.pumps_active,
            'power_level': result.power_level,
            'expected_savings': result.expected_savings,
            'savings_percent': result.savings_percent,
            'actions': result.actions
        }
        
        total_savings += result.expected_savings
    
    return {
        'stations': results,
        'total_savings': total_savings,
        'timestamp': pd.Timestamp.now()
    }


if __name__ == "__main__":
    # Test
    station_config = {
        'station_id': 'OUG_ZOG',
        'name': 'Station de Zogona',
        'pump_count': 4,
        'max_power_kw': 1200,
        'max_flow_m3h': 4500
    }
    
    optimizer = SmartPumpOptimizer(station_config)
    
    print("🎯 Test Optimisation Intelligente")
    print("="*60)
    
    # Test différentes heures
    test_cases = [
        (2, 2500, "Heure creuse - Faible demande"),
        (8, 3800, "Heure normale - Demande moyenne"),
        (20, 4200, "Heure pleine - Forte demande")
    ]
    
    for hour, demand, description in test_cases:
        print(f"\n📊 {description}")
        print(f"   Heure: {hour}h | Demande: {demand} m³/h")
        
        result = optimizer.optimize_for_period(hour, demand)
        
        print(f"   ✅ Optimisation:")
        print(f"      Pompes: {result.pumps_active}/{station_config['pump_count']}")
        print(f"      Puissance: {result.power_level*100:.0f}%")
        print(f"      Coût: {result.expected_cost:.0f} FCFA")
        print(f"      Économie: {result.expected_savings:.0f} FCFA ({result.savings_percent:.1f}%)")
        print(f"      Actions: {len(result.actions)}")
