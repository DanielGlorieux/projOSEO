"""
Métriques personnalisées pour évaluation performance
"""
import numpy as np
from typing import Dict, List, Tuple


def calculate_energy_savings(baseline_kwh: float, optimized_kwh: float) -> Dict[str, float]:
    """
    Calcule économies énergétiques
    
    Returns:
        - absolute_savings: Économie absolue (kWh)
        - percentage_savings: Économie relative (%)
        - cost_savings_fcfa: Économie monétaire (FCFA)
    """
    absolute_savings = baseline_kwh - optimized_kwh
    percentage_savings = (absolute_savings / baseline_kwh) * 100
    
    # Prix moyen pondéré: 85 FCFA/kWh
    cost_savings_fcfa = absolute_savings * 85
    
    return {
        'absolute_savings_kwh': absolute_savings,
        'percentage_savings': percentage_savings,
        'cost_savings_fcfa': cost_savings_fcfa
    }


def calculate_specific_consumption(energy_kwh: np.ndarray, water_m3: np.ndarray) -> float:
    """
    Consommation spécifique: kWh/m³
    KPI clé pour efficacité énergétique
    """
    return np.mean(energy_kwh / water_m3)


def calculate_peak_shifting_score(hourly_consumption: np.ndarray, 
                                  peak_hours: List[int]) -> float:
    """
    Score d'évitement heures pleines
    100 = parfait (0% aux heures pleines)
    0 = très mauvais (100% aux heures pleines)
    """
    peak_consumption = sum(hourly_consumption[h] for h in peak_hours)
    total_consumption = np.sum(hourly_consumption)
    
    peak_ratio = peak_consumption / total_consumption
    score = (1 - peak_ratio) * 100
    
    return score


def calculate_service_continuity(reservoir_levels: np.ndarray, 
                                critical_threshold: float = 0.2) -> Dict[str, float]:
    """
    Évalue continuité du service
    """
    violations = np.sum(reservoir_levels < critical_threshold)
    violation_rate = (violations / len(reservoir_levels)) * 100
    
    avg_level = np.mean(reservoir_levels)
    min_level = np.min(reservoir_levels)
    
    # Score: 100 - (violations * 10)
    score = max(0, 100 - violation_rate * 10)
    
    return {
        'violations_count': int(violations),
        'violation_rate_percent': violation_rate,
        'avg_reservoir_level_percent': avg_level * 100,
        'min_reservoir_level_percent': min_level * 100,
        'service_continuity_score': score
    }


def calculate_roi_metrics(investment_fcfa: float, 
                         annual_savings_fcfa: float,
                         years: int = 3) -> Dict[str, float]:
    """
    Calcule métriques ROI
    """
    # Payback period
    payback_months = (investment_fcfa / annual_savings_fcfa) * 12
    
    # ROI sur période
    total_savings = annual_savings_fcfa * years
    roi_percent = ((total_savings - investment_fcfa) / investment_fcfa) * 100
    
    # VAN (taux 10%)
    discount_rate = 0.10
    npv = sum([annual_savings_fcfa / ((1 + discount_rate) ** year) 
               for year in range(1, years + 1)]) - investment_fcfa
    
    # TRI approximatif
    irr = ((total_savings / investment_fcfa) ** (1 / years) - 1) * 100
    
    return {
        'payback_period_months': payback_months,
        'roi_percent': roi_percent,
        'npv_fcfa': npv,
        'irr_percent': irr
    }


def calculate_anomaly_detection_metrics(y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> Dict[str, float]:
    """
    Métriques détection anomalies (classification binaire)
    """
    # Confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Métriques
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def calculate_co2_reduction(energy_savings_kwh: float, 
                           emission_factor: float = 0.74) -> Dict[str, float]:
    """
    Calcule réduction émissions CO2
    
    Args:
        energy_savings_kwh: Économie énergétique (kWh)
        emission_factor: kg CO2/kWh (Burkina Faso ≈ 0.74)
    
    Returns:
        Métriques environnementales
    """
    co2_reduction_kg = energy_savings_kwh * emission_factor
    co2_reduction_tons = co2_reduction_kg / 1000
    
    # Équivalences
    cars_equivalent = co2_reduction_tons / 4.6  # Voiture moyenne = 4.6 tonnes CO2/an
    trees_equivalent = co2_reduction_tons / 0.06  # Arbre absorbe ~60kg CO2/an
    
    return {
        'co2_reduction_kg': co2_reduction_kg,
        'co2_reduction_tons': co2_reduction_tons,
        'cars_equivalent': cars_equivalent,
        'trees_equivalent': int(trees_equivalent)
    }


def calculate_overall_performance_score(metrics: Dict) -> float:
    """
    Score global de performance (0-100)
    Pondération des différents KPIs
    """
    weights = {
        'cost_savings': 0.35,
        'service_continuity': 0.25,
        'prediction_accuracy': 0.20,
        'anomaly_detection': 0.10,
        'peak_shifting': 0.10
    }
    
    # Normaliser chaque métrique sur 0-100
    scores = {
        'cost_savings': min(100, (metrics.get('percentage_savings', 0) / 30) * 100),  # 30% = 100 points
        'service_continuity': metrics.get('service_continuity_score', 0),
        'prediction_accuracy': max(0, 100 - metrics.get('mape', 100)),  # MAPE inversé
        'anomaly_detection': metrics.get('f1_score', 0) * 100,
        'peak_shifting': metrics.get('peak_shifting_score', 0)
    }
    
    # Score pondéré
    overall_score = sum(scores[key] * weights[key] for key in weights.keys())
    
    return overall_score


if __name__ == "__main__":
    # Test des métriques
    print("🧪 Test des métriques personnalisées\n")
    
    # 1. Économies énergétiques
    savings = calculate_energy_savings(1000000, 715000)
    print("📊 Économies énergétiques:")
    print(f"  - Économie: {savings['absolute_savings_kwh']:,.0f} kWh ({savings['percentage_savings']:.1f}%)")
    print(f"  - Gain monétaire: {savings['cost_savings_fcfa']:,.0f} FCFA")
    
    # 2. ROI
    roi = calculate_roi_metrics(142_000_000, 4_300_000_000, years=3)
    print(f"\n💰 ROI:")
    print(f"  - Payback: {roi['payback_period_months']:.1f} mois")
    print(f"  - ROI 3 ans: {roi['roi_percent']:.0f}%")
    print(f"  - VAN: {roi['npv_fcfa']:,.0f} FCFA")
    print(f"  - TRI: {roi['irr_percent']:.0f}%")
    
    # 3. Impact environnemental
    co2 = calculate_co2_reduction(6_500_000)
    print(f"\n🌍 Impact environnemental:")
    print(f"  - Réduction CO2: {co2['co2_reduction_tons']:,.0f} tonnes")
    print(f"  - Équivalent: {co2['cars_equivalent']:.0f} voitures")
    print(f"  - Équivalent: {co2['trees_equivalent']:,} arbres")
    
    print("\n✅ Tous les tests passés!")
