"""
Utilitaires de prétraitement des données
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


def create_sequences(data: np.ndarray, seq_length: int, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences pour modèles temporels (LSTM, etc.)
    
    Args:
        data: Données temporelles
        seq_length: Longueur séquence d'entrée (ex: 168h = 1 semaine)
        forecast_horizon: Horizon de prédiction (ex: 24h)
    
    Returns:
        X, y: Séquences d'entrée et cibles
    """
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])
    return np.array(X), np.array(y)


def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Ajoute des features temporelles"""
    df = df.copy()
    
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['quarter'] = df[timestamp_col].dt.quarter
    df['week_of_year'] = df[timestamp_col].dt.isocalendar().week
    
    # Features cycliques
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Flags temporels
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].isin(range(18, 23)).astype(int)
    df['is_off_peak'] = df['hour'].isin(list(range(0, 6)) + [23]).astype(int)
    
    return df


def add_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Ajoute features de lag (valeurs passées)"""
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, columns: List[str], 
                        windows: List[int] = [24, 168, 720]) -> pd.DataFrame:
    """Ajoute statistiques rolling (moyennes mobiles, etc.)"""
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Détecte outliers avec méthode IQR"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """Gère les valeurs manquantes"""
    df = df.copy()
    
    if method == 'interpolate':
        df = df.interpolate(method='time')
    elif method == 'forward_fill':
        df = df.fillna(method='ffill')
    elif method == 'mean':
        df = df.fillna(df.mean())
    
    return df


class DataPreprocessor:
    """Classe pour preprocessing complet avec target scaler"""
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scalers = {}
        self.target_scaler = None  # NOUVEAU: Scaler pour target
        self.feature_names = []
        
    def fit_transform(self, df: pd.DataFrame, 
                     numerical_cols: List[str],
                     target_col: str = 'energy_consumption_kwh',
                     add_time_feats: bool = True,
                     add_lags: bool = True,
                     add_rolling: bool = True) -> pd.DataFrame:
        """
        Preprocessing complet avec fit + target scaling
        """
        df = df.copy()
        
        # Features temporelles
        if add_time_feats and 'timestamp' in df.columns:
            df = add_time_features(df)
        
        # Features lag
        if add_lags:
            df = add_lag_features(
                df, 
                columns=['energy_consumption_kwh', 'water_demand_m3', 'power_kw'],
                lags=[1, 24, 168]
            )
        
        # Features rolling
        if add_rolling:
            df = add_rolling_features(
                df,
                columns=['energy_consumption_kwh', 'water_demand_m3'],
                windows=[24, 168]
            )
        
        # Normalisation features
        for col in numerical_cols:
            if col in df.columns:
                if self.scaler_type == 'standard':
                    scaler = StandardScaler()
                else:
                    scaler = MinMaxScaler()
                
                df[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
        
        # NOUVEAU: Normaliser aussi le target
        if target_col in df.columns:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            df[f'{target_col}_scaled'] = self.target_scaler.fit_transform(df[[target_col]])
        
        # Supprimer NaN créés par lag/rolling
        df = df.dropna()
        
        self.feature_names = df.columns.tolist()
        
        return df
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        NOUVEAU: Dénormalise les prédictions
        """
        if self.target_scaler is None:
            return y_scaled
        
        # Reshape si nécessaire
        original_shape = y_scaled.shape
        y_reshaped = y_scaled.reshape(-1, 1)
        y_original = self.target_scaler.inverse_transform(y_reshaped)
        
        return y_original.reshape(original_shape)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transformation sans fit (pour données test/production)
        """
        df = df.copy()
        
        # Appliquer mêmes transformations
        if 'timestamp' in df.columns:
            df = add_time_features(df)
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[f'{col}_scaled'] = scaler.transform(df[[col]])
        
        return df
    
    def save(self, filepath: str):
        """Sauvegarde le preprocessor"""
        joblib.dump({
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'scaler_type': self.scaler_type
        }, filepath)
    
    def load(self, filepath: str):
        """Charge le preprocessor"""
        data = joblib.load(filepath)
        self.scalers = data['scalers']
        self.feature_names = data['feature_names']
        self.scaler_type = data['scaler_type']


def split_train_val_test(df: pd.DataFrame, 
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporel (important pour séries temporelles)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences pour LSTM
    
    Args:
        X: Features (n_samples, n_features)
        y: Target values (n_samples,)
        seq_length: Longueur séquence entrée
        forecast_horizon: Horizon prédiction
    
    Returns:
        X_seq: (n_sequences, seq_length, n_features)
        y_seq: (n_sequences, forecast_horizon)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length - forecast_horizon + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length:i + seq_length + forecast_horizon])
    
    return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    # Test preprocessing
    print("Test du preprocessing...")
    
    # Charger données exemple
    from pathlib import Path
    data_file = Path(__file__).parent.parent / "data" / "raw" / "OUG_ZOG_historical.csv"
    
    if data_file.exists():
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        print(f"✓ Données chargées: {len(df)} lignes")
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(
            df,
            numerical_cols=['energy_consumption_kwh', 'water_demand_m3', 'power_kw']
        )
        
        print(f"✓ Preprocessing terminé: {len(df_processed)} lignes, {len(df_processed.columns)} colonnes")
        print(f"\nNouvelles features: {df_processed.columns.tolist()[:20]}...")
        
        # Split
        train, val, test = split_train_val_test(df_processed)
        print(f"\n✓ Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    else:
        print("⚠️  Génération des données d'abord avec: python data/synthetic_generator.py")
