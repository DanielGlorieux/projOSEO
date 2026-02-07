"""
Configuration centrale pour le projet ONEA Smart Energy Optimizer
"""
import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import yaml

# Chemins de base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Créer les dossiers nécessaires
for dir_path in [DATA_DIR / "raw", DATA_DIR / "processed", MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class StationConfig:
    """Configuration d'une station de pompage"""
    station_id: str
    name: str
    location: str
    pump_count: int
    max_power_kw: float
    reservoir_capacity_m3: float
    latitude: float
    longitude: float


@dataclass
class EnergyPricingConfig:
    """Configuration tarifaire SONABEL (exemple réaliste Burkina Faso)"""
    # Heures pleines: 18h-23h (100 FCFA/kWh)
    peak_hours: List[int]
    peak_price_fcfa: float
    
    # Heures creuses: 23h-6h (65 FCFA/kWh)
    off_peak_hours: List[int]
    off_peak_price_fcfa: float
    
    # Heures normales: 6h-18h (85 FCFA/kWh)
    normal_hours: List[int]
    normal_price_fcfa: float
    
    # Pénalités
    power_factor_penalty_threshold: float  # < 0.85
    penalty_rate: float  # +15% sur facture


# Configuration des stations ONEA (exemples réalistes)
STATIONS: List[StationConfig] = [
    StationConfig(
        station_id="OUG_ZOG",
        name="Station de Zogona",
        location="Ouagadougou",
        pump_count=4,
        max_power_kw=850.0,
        reservoir_capacity_m3=5000.0,
        latitude=12.3714,
        longitude=-1.5197
    ),
    StationConfig(
        station_id="OUG_PIS",
        name="Station de Pissy",
        location="Ouagadougou",
        pump_count=6,
        max_power_kw=1200.0,
        reservoir_capacity_m3=8000.0,
        latitude=12.3389,
        longitude=-1.5439
    ),
    StationConfig(
        station_id="BOBO_KUA",
        name="Station de Kuinima",
        location="Bobo-Dioulasso",
        pump_count=5,
        max_power_kw=950.0,
        reservoir_capacity_m3=6000.0,
        latitude=11.1773,
        longitude=-4.2970
    ),
    StationConfig(
        station_id="OUG_NAB",
        name="Station de Nabitenga",
        location="Ouagadougou",
        pump_count=3,
        max_power_kw=600.0,
        reservoir_capacity_m3=3500.0,
        latitude=12.4048,
        longitude=-1.4755
    ),
    StationConfig(
        station_id="BOBO_DAR",
        name="Station de Darsalamy",
        location="Bobo-Dioulasso",
        pump_count=4,
        max_power_kw=750.0,
        reservoir_capacity_m3=4500.0,
        latitude=11.1886,
        longitude=-4.3089
    ),
]

# Configuration tarifaire
ENERGY_PRICING = EnergyPricingConfig(
    peak_hours=list(range(18, 23)),
    peak_price_fcfa=100.0,
    off_peak_hours=list(range(23, 24)) + list(range(0, 6)),
    off_peak_price_fcfa=65.0,
    normal_hours=list(range(6, 18)),
    normal_price_fcfa=85.0,
    power_factor_penalty_threshold=0.85,
    penalty_rate=0.15
)

# Hyperparamètres Machine Learning - AUTO-OPTIMISÉS GPU/CPU
import torch

# Détection automatique GPU
IS_GPU_AVAILABLE = torch.cuda.is_available()
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if IS_GPU_AVAILABLE else 0

ML_CONFIG = {
    "forecasting": {
        "lstm": {
            "hidden_size": 256 if IS_GPU_AVAILABLE else 192,  # Plus grand sur GPU
            "num_layers": 4 if IS_GPU_AVAILABLE else 3,     # Plus profond sur GPU
            "dropout": 0.5,
            "learning_rate": 0.001 if IS_GPU_AVAILABLE else 0.0005,  # Plus agressif sur GPU
            "epochs": 150 if IS_GPU_AVAILABLE else 80,       # Plus d'epochs sur GPU
            "batch_size": 256 if IS_GPU_AVAILABLE else 64,   # Batch plus grand sur GPU
            "sequence_length": 168,  # 1 semaine en heures
            "num_workers": 4 if IS_GPU_AVAILABLE else 0,     # Workers pour DataLoader
            "pin_memory": IS_GPU_AVAILABLE,                  # Pin memory si GPU
            "use_amp": IS_GPU_AVAILABLE                      # Mixed precision si GPU
        },
        "prophet": {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10,
            "seasonality_mode": "multiplicative",
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": True
        },
        "xgboost": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
    },
    "optimization": {
        "rl_agent": {
            "algorithm": "PPO",
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 128 if IS_GPU_AVAILABLE else 64,   # Plus grand sur GPU
            "n_epochs": 15 if IS_GPU_AVAILABLE else 10,      # Plus d'epochs sur GPU
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "total_timesteps": 500000 if IS_GPU_AVAILABLE else 200000,  # Plus de steps sur GPU
            "device": "cuda" if IS_GPU_AVAILABLE else "cpu"
        },
        "genetic_algorithm": {
            "population_size": 100,
            "generations": 50,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8
        }
    },
    "anomaly_detection": {
        "isolation_forest": {
            "n_estimators": 200,
            "contamination": 0.05,
            "max_samples": "auto"
        },
        "autoencoder": {
            "encoding_dim": 32,
            "epochs": 50,
            "batch_size": 256,
            "threshold_percentile": 95
        }
    }
}

# Configuration Digital Twin
DIGITAL_TWIN_CONFIG = {
    "simulation_timestep_minutes": 15,
    "prediction_horizon_hours": 48,
    "reservoir_alert_low_percent": 20.0,
    "reservoir_alert_high_percent": 95.0,
    "pressure_min_bar": 2.5,
    "pressure_max_bar": 6.0,
    "pump_efficiency_threshold": 0.70,
    "leak_detection_threshold_m3h": 50.0
}

# Configuration API
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "cors_origins": ["*"],
    "rate_limit": "100/minute"
}

# Configuration Dashboard
DASHBOARD_CONFIG = {
    "title": "ONEA Smart Energy Optimizer",
    "layout": "wide",
    "theme": "dark",
    "refresh_interval_seconds": 30,
    "max_data_points_display": 10000
}

# Objectifs d'optimisation
OPTIMIZATION_TARGETS = {
    "energy_cost_reduction_percent": 28.0,
    "co2_reduction_percent": 30.0,
    "penalty_reduction_percent": 90.0,
    "anomaly_detection_f1_score": 0.95,
    "forecast_mape_threshold": 6.0
}

# Métriques de performance
PERFORMANCE_METRICS = [
    "energy_consumption_kwh",
    "energy_cost_fcfa",
    "water_production_m3",
    "specific_consumption_kwh_m3",
    "power_factor",
    "pump_efficiency_percent",
    "reservoir_level_percent",
    "pressure_bar",
    "anomaly_score"
]


def get_station_by_id(station_id: str) -> StationConfig:
    """Récupère une station par son ID"""
    for station in STATIONS:
        if station.station_id == station_id:
            return station
    raise ValueError(f"Station {station_id} non trouvée")


def get_energy_price(hour: int) -> float:
    """Retourne le prix de l'énergie selon l'heure"""
    if hour in ENERGY_PRICING.peak_hours:
        return ENERGY_PRICING.peak_price_fcfa
    elif hour in ENERGY_PRICING.off_peak_hours:
        return ENERGY_PRICING.off_peak_price_fcfa
    else:
        return ENERGY_PRICING.normal_price_fcfa


def calculate_cost_with_penalties(kwh: float, power_factor: float, hour: int) -> float:
    """Calcule le coût énergétique avec pénalités éventuelles"""
    base_cost = kwh * get_energy_price(hour)
    
    if power_factor < ENERGY_PRICING.power_factor_penalty_threshold:
        penalty = base_cost * ENERGY_PRICING.penalty_rate
        return base_cost + penalty
    
    return base_cost


# Configuration logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "onea_optimizer.log"),
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}


if __name__ == "__main__":
    # Test configuration
    print("=== Configuration ONEA Smart Energy Optimizer ===")
    print(f"\nNombre de stations: {len(STATIONS)}")
    for station in STATIONS:
        print(f"  - {station.name} ({station.station_id}): {station.pump_count} pompes, {station.max_power_kw}kW")
    
    print(f"\nTarification:")
    print(f"  - Heures creuses (23h-6h): {ENERGY_PRICING.off_peak_price_fcfa} FCFA/kWh")
    print(f"  - Heures normales (6h-18h): {ENERGY_PRICING.normal_price_fcfa} FCFA/kWh")
    print(f"  - Heures pleines (18h-23h): {ENERGY_PRICING.peak_price_fcfa} FCFA/kWh")
    
    print(f"\nObjectifs d'optimisation:")
    for key, value in OPTIMIZATION_TARGETS.items():
        print(f"  - {key}: {value}")
