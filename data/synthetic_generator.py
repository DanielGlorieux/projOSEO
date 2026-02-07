"""
Générateur de données synthétiques réalistes pour l'ONEA
Simule 2 ans de données historiques avec patterns réalistes
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import STATIONS, ENERGY_PRICING, get_energy_price

np.random.seed(42)


class ONEADataGenerator:
    """Génère des données synthétiques réalistes pour les stations ONEA"""
    
    def __init__(self, start_date="2020-01-01", periods=24*365*6):
        """
        Args:
            start_date: Date de début
            periods: Nombre d'heures à générer (6 ans par défaut pour meilleur apprentissage)
        """
        self.start_date = pd.to_datetime(start_date)
        self.periods = periods
        self.timestamps = pd.date_range(start=self.start_date, periods=periods, freq='H')
        
    def generate_water_demand(self, station_capacity_m3: float, station_location: str) -> np.ndarray:
        """
        Génère la demande en eau avec patterns réalistes
        - Saisonnalité annuelle (saison sèche/pluies)
        - Saisonnalité hebdomadaire
        - Patterns journaliers (pics matin/soir)
        - Tendance croissance
        - Bruit aléatoire
        """
        t = np.arange(self.periods)
        base_demand = station_capacity_m3 * 0.6  # 60% de capacité moyenne
        
        # Tendance croissance (+2% par an)
        trend = base_demand * (1 + 0.02 * t / (365*24))
        
        # Saisonnalité annuelle (saison sèche Nov-Mai: +20%, saison pluies Jun-Oct: -10%)
        annual_cycle = 0.15 * np.sin(2 * np.pi * t / (365*24) - np.pi/2)
        
        # Saisonnalité hebdomadaire (moins de demande week-end)
        weekly_cycle = -0.05 * np.sin(2 * np.pi * t / (7*24))
        
        # Pattern journalier (pics à 7h et 19h)
        hour_of_day = t % 24
        daily_cycle = (
            0.2 * np.exp(-((hour_of_day - 7)**2) / 8) +  # Pic matin
            0.25 * np.exp(-((hour_of_day - 19)**2) / 8) -  # Pic soir
            0.15 * np.exp(-((hour_of_day - 2)**2) / 8)    # Creux nuit
        )
        
        # Effet géographique (Ouaga > Bobo)
        if station_location == "Ouagadougou":
            location_factor = 1.15
        else:
            location_factor = 1.0
            
        # Bruit aléatoire réaliste (5-8% de variance)
        noise = np.random.normal(0, 0.06, self.periods)
        
        demand = trend * (1 + annual_cycle + weekly_cycle + daily_cycle + noise) * location_factor
        
        # Ajouter événements exceptionnels réalistes (pannes, travaux, pics)
        # Plus d'événements sur 6 ans
        num_events = np.random.randint(15, 40)
        for _ in range(num_events):
            event_start = np.random.randint(0, self.periods - 48)
            event_duration = np.random.randint(4, 48)
            event_type = np.random.choice(['drop', 'spike'])
            if event_type == 'drop':
                demand[event_start:event_start+event_duration] *= np.random.uniform(0.4, 0.7)
            else:
                demand[event_start:event_start+event_duration] *= np.random.uniform(1.2, 1.5)
        
        return np.maximum(demand, station_capacity_m3 * 0.1)  # Minimum 10% capacité
    
    def generate_energy_consumption(self, water_demand: np.ndarray, 
                                    num_pumps: int, 
                                    max_power_kw: float) -> tuple:
        """
        Génère consommation énergétique basée sur demande eau
        Retourne: (energy_kwh, power_kw, pump_status, efficiency)
        """
        # Consommation spécifique de base: 0.35-0.55 kWh/m3 (typique Afrique subsaharienne)
        specific_consumption_base = np.random.uniform(0.35, 0.55, self.periods)
        
        # Inefficiences:
        # - Heures pleines: pompes moins efficaces (surtension)
        # - Vieillissement équipement: dégradation progressive
        # - Facteur de charge sous-optimal
        hour_of_day = np.arange(self.periods) % 24
        peak_inefficiency = np.where(np.isin(hour_of_day, ENERGY_PRICING.peak_hours), 1.15, 1.0)
        
        # Dégradation progressive efficacité (-0.5% par trimestre)
        degradation = 1 + (np.arange(self.periods) / (365*24)) * 0.02
        
        # Inefficience facteur de charge (pompes surdimensionnées)
        load_factor = water_demand / water_demand.max()
        load_inefficiency = 1 + 0.1 * (1 - load_factor)  # Moins efficace à charge partielle
        
        specific_consumption = (specific_consumption_base * 
                               peak_inefficiency * 
                               degradation * 
                               load_inefficiency)
        
        # Consommation énergétique horaire
        energy_kwh = water_demand * specific_consumption
        
        # Puissance instantanée (avec variations)
        power_kw = energy_kwh + np.random.normal(0, energy_kwh * 0.05)
        power_kw = np.clip(power_kw, max_power_kw * 0.1, max_power_kw * 0.95)
        
        # Statut pompes (nombre actif)
        pumps_active = np.ceil(power_kw / (max_power_kw / num_pumps)).astype(int)
        pumps_active = np.clip(pumps_active, 1, num_pumps)
        
        # Efficacité pompes (70-90%)
        efficiency = np.random.uniform(0.70, 0.90, self.periods)
        efficiency = efficiency * (1 - 0.01 * (np.arange(self.periods) / (365*24)))  # Dégradation
        efficiency = np.clip(efficiency, 0.65, 0.92)
        
        return energy_kwh, power_kw, pumps_active, efficiency
    
    def generate_reservoir_levels(self, water_demand: np.ndarray, 
                                  capacity_m3: float) -> np.ndarray:
        """Génère les niveaux de réservoir"""
        # Modèle simplifié: niveau = production - consommation
        production = water_demand * 1.05  # Production légèrement > demande
        
        level = np.zeros(self.periods)
        level[0] = capacity_m3 * 0.7  # Début à 70%
        
        for i in range(1, self.periods):
            level[i] = level[i-1] + production[i] - water_demand[i]
            level[i] = np.clip(level[i], capacity_m3 * 0.15, capacity_m3 * 0.98)
        
        # Ajouter variations aléatoires
        level += np.random.normal(0, capacity_m3 * 0.02, self.periods)
        level = np.clip(level, capacity_m3 * 0.1, capacity_m3)
        
        return level
    
    def generate_power_factor(self) -> np.ndarray:
        """Génère le facteur de puissance (cos φ)"""
        # Facteur moyen: 0.82-0.88 (souvent sous optimal)
        base_pf = np.random.uniform(0.82, 0.88, self.periods)
        
        # Variations selon charge
        hour_of_day = np.arange(self.periods) % 24
        # Moins bon aux heures de pointe
        peak_penalty = np.where(np.isin(hour_of_day, ENERGY_PRICING.peak_hours), -0.05, 0)
        
        power_factor = base_pf + peak_penalty + np.random.normal(0, 0.02, self.periods)
        return np.clip(power_factor, 0.75, 0.95)
    
    def add_anomalies(self, data: pd.DataFrame, anomaly_rate=0.03) -> pd.DataFrame:
        """Ajoute des anomalies réalistes (fuites, pannes, surconsommations)"""
        num_anomalies = int(len(data) * anomaly_rate)
        anomaly_indices = np.random.choice(len(data), num_anomalies, replace=False)
        
        data['anomaly'] = 0
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['leak', 'overconsumption', 'pump_failure'], 
                                           p=[0.4, 0.4, 0.2])
            duration = np.random.randint(2, 24)  # 2-24h
            end_idx = min(idx + duration, len(data))
            
            if anomaly_type == 'leak':
                # Fuite: demande constante mais consommation élevée
                data.loc[idx:end_idx, 'energy_consumption_kwh'] *= np.random.uniform(1.3, 1.8)
                data.loc[idx:end_idx, 'specific_consumption_kwh_m3'] *= np.random.uniform(1.3, 1.8)
                data.loc[idx:end_idx, 'anomaly'] = 1
                
            elif anomaly_type == 'overconsumption':
                # Surconsommation: inefficacité temporaire
                data.loc[idx:end_idx, 'energy_consumption_kwh'] *= np.random.uniform(1.2, 1.5)
                data.loc[idx:end_idx, 'pump_efficiency'] *= 0.7
                data.loc[idx:end_idx, 'anomaly'] = 2
                
            elif anomaly_type == 'pump_failure':
                # Panne pompe: compensation par autres pompes
                data.loc[idx:end_idx, 'pumps_active'] = data.loc[idx:end_idx, 'pumps_active'].clip(upper=1)
                data.loc[idx:end_idx, 'power_kw'] *= np.random.uniform(1.1, 1.3)
                data.loc[idx:end_idx, 'anomaly'] = 3
        
        return data
    
    def generate_station_data(self, station_config) -> pd.DataFrame:
        """Génère dataset complet pour une station"""
        print(f"Génération données pour {station_config.name}...")
        
        # Demande en eau
        water_demand = self.generate_water_demand(
            station_config.reservoir_capacity_m3,
            station_config.location
        )
        
        # Consommation énergétique
        energy_kwh, power_kw, pumps_active, efficiency = self.generate_energy_consumption(
            water_demand,
            station_config.pump_count,
            station_config.max_power_kw
        )
        
        # Niveaux réservoir
        reservoir_level = self.generate_reservoir_levels(
            water_demand,
            station_config.reservoir_capacity_m3
        )
        
        # Facteur de puissance
        power_factor = self.generate_power_factor()
        
        # Créer DataFrame
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'station_id': station_config.station_id,
            'station_name': station_config.name,
            'location': station_config.location,
            'water_demand_m3': water_demand,
            'water_production_m3': water_demand * 1.05,
            'energy_consumption_kwh': energy_kwh,
            'power_kw': power_kw,
            'pumps_active': pumps_active,
            'pumps_total': station_config.pump_count,
            'pump_efficiency': efficiency,
            'reservoir_level_m3': reservoir_level,
            'reservoir_capacity_m3': station_config.reservoir_capacity_m3,
            'reservoir_level_percent': (reservoir_level / station_config.reservoir_capacity_m3) * 100,
            'power_factor': power_factor,
            'specific_consumption_kwh_m3': energy_kwh / water_demand,
        })
        
        # Calculer coûts énergétiques
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['energy_price_fcfa_kwh'] = df['hour'].apply(get_energy_price)
        
        # Coût avec pénalités
        df['energy_cost_fcfa'] = df.apply(
            lambda row: row['energy_consumption_kwh'] * row['energy_price_fcfa_kwh'] * 
                       (1.15 if row['power_factor'] < 0.85 else 1.0),
            axis=1
        )
        
        df['has_penalty'] = (df['power_factor'] < 0.85).astype(int)
        
        # Ajouter anomalies
        df = self.add_anomalies(df)
        
        return df
    
    def generate_all_stations(self, output_dir: Path) -> dict:
        """Génère et sauvegarde données pour toutes les stations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_data = {}
        combined_data = []
        
        for station in STATIONS:
            df = self.generate_station_data(station)
            
            # Sauvegarder individuellement
            station_file = output_dir / f"{station.station_id}_historical.csv"
            df.to_csv(station_file, index=False)
            print(f"  ✓ Sauvegardé: {station_file}")
            
            all_data[station.station_id] = df
            combined_data.append(df)
        
        # Dataset combiné
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_file = output_dir / "all_stations_historical.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✓ Dataset combiné: {combined_file}")
        
        # Statistiques
        self.print_statistics(combined_df)
        
        return all_data
    
    def print_statistics(self, df: pd.DataFrame):
        """Affiche statistiques du dataset généré"""
        print("\n" + "="*60)
        print("STATISTIQUES DU DATASET GÉNÉRÉ")
        print("="*60)
        
        print(f"\n📊 Taille dataset:")
        print(f"  - Nombre total de lignes: {len(df):,}")
        print(f"  - Période: {df['timestamp'].min()} à {df['timestamp'].max()}")
        print(f"  - Durée: {(df['timestamp'].max() - df['timestamp'].min()).days} jours")
        
        print(f"\n⚡ Énergie:")
        print(f"  - Consommation totale: {df['energy_consumption_kwh'].sum():,.0f} kWh")
        print(f"  - Consommation moyenne horaire: {df['energy_consumption_kwh'].mean():.1f} kWh")
        print(f"  - Coût total: {df['energy_cost_fcfa'].sum():,.0f} FCFA")
        print(f"  - Coût moyen mensuel: {df['energy_cost_fcfa'].sum() / 24:,.0f} FCFA")
        
        print(f"\n💧 Eau:")
        print(f"  - Production totale: {df['water_production_m3'].sum():,.0f} m³")
        print(f"  - Consommation spécifique moyenne: {df['specific_consumption_kwh_m3'].mean():.3f} kWh/m³")
        
        print(f"\n⚠️ Anomalies:")
        anomaly_count = (df['anomaly'] > 0).sum()
        print(f"  - Nombre d'anomalies: {anomaly_count:,} ({anomaly_count/len(df)*100:.2f}%)")
        print(f"    • Fuites (type 1): {(df['anomaly'] == 1).sum():,}")
        print(f"    • Surconsommations (type 2): {(df['anomaly'] == 2).sum():,}")
        print(f"    • Pannes pompes (type 3): {(df['anomaly'] == 3).sum():,}")
        
        print(f"\n🏭 Stations:")
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id]
            print(f"  - {station_id}: {len(station_data):,} observations")
        
        print(f"\n💰 Pénalités facteur de puissance:")
        penalty_count = df['has_penalty'].sum()
        penalty_cost = df[df['has_penalty'] == 1]['energy_cost_fcfa'].sum() * 0.15 / 1.15
        print(f"  - Occurrences: {penalty_count:,} ({penalty_count/len(df)*100:.2f}%)")
        print(f"  - Coût total pénalités: {penalty_cost:,.0f} FCFA")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    print("🚀 Génération des données ONEA Smart Energy Optimizer\n")
    
    # Générer 6 ans de données horaires (améliore SIGNIFICATIVEMENT l'apprentissage)
    generator = ONEADataGenerator(
        start_date="2020-01-01",
        periods=24 * 365 * 6
    )
    
    # Créer dossier output
    output_dir = Path(__file__).parent / "raw"
    
    # Générer toutes les stations
    all_data = generator.generate_all_stations(output_dir)
    
    print("\n✅ Génération terminée avec succès!")
    print(f"\n📁 Fichiers générés dans: {output_dir}")
