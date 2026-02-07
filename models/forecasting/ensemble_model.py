"""
Modle de prdiction optimis avec XGBoost et LightGBM
Remplacement du LSTM pour entrainement plus rapide sur CPU
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path
import joblib


class EnsembleForecastModel:
    """Modle ensemble XGBoost + LightGBM pour prdiction rapide"""
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.feature_cols = None
        
    def create_features(self, df):
        """Cration features temporelles optimises"""
        df = df.copy()
        
        # Features temporelles
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Features cycliques
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Features lag
        df['demand_lag_1'] = df['water_demand_m3'].shift(1)
        df['demand_lag_24'] = df['water_demand_m3'].shift(24)
        df['demand_lag_168'] = df['water_demand_m3'].shift(168)
        
        # Rolling features
        df['demand_roll_mean_24'] = df['water_demand_m3'].rolling(window=24, min_periods=1).mean()
        df['demand_roll_std_24'] = df['water_demand_m3'].rolling(window=24, min_periods=1).std()
        df['demand_roll_max_24'] = df['water_demand_m3'].rolling(window=24, min_periods=1).max()
        df['demand_roll_min_24'] = df['water_demand_m3'].rolling(window=24, min_periods=1).min()
        
        # Tarification
        df['is_peak_hour'] = df['hour'].apply(lambda h: 1 if 18 <= h < 23 else 0)
        df['is_off_peak'] = df['hour'].apply(lambda h: 1 if h >= 23 or h < 6 else 0)
        
        return df
    
    def train(self, df, target_col='water_demand_m3'):
        """Entrainement rapide du modle ensemble"""
        print(" Cration des features...")
        df = self.create_features(df)
        df = df.dropna()
        
        # Prparation donnes
        self.feature_cols = [
            'hour', 'day_of_week', 'month', 'day_of_month', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
            'demand_roll_mean_24', 'demand_roll_std_24',
            'demand_roll_max_24', 'demand_roll_min_24',
            'is_peak_hour', 'is_off_peak'
        ]
        
        X = df[self.feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        print(f" Donnes: Train={len(X_train)}, Test={len(X_test)}")
        
        # XGBoost
        print(" Entrainement XGBoost...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Plus rapide sur CPU
        )
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # LightGBM
        print(" Entrainement LightGBM...")
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)]
        )
        
        # valuation
        print("\n valuation du modle ensemble...")
        y_pred_xgb = self.xgb_model.predict(X_test)
        y_pred_lgb = self.lgb_model.predict(X_test)
        y_pred = (y_pred_xgb + y_pred_lgb) / 2  # Moyenne ensemble
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"\n Rsultats:")
        print(f"   MAE:  {mae:.2f} m/h")
        print(f"   RMSE: {rmse:.2f} m/h")
        print(f"   R:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def predict(self, df, horizon_hours=24):
        """Prdiction rapide"""
        df = self.create_features(df)
        df = df.dropna()
        
        X = df[self.feature_cols].tail(1)
        
        predictions = []
        for h in range(horizon_hours):
            pred_xgb = self.xgb_model.predict(X)[0]
            pred_lgb = self.lgb_model.predict(X)[0]
            pred = (pred_xgb + pred_lgb) / 2
            predictions.append(pred)
            
            # Update features for next step
            X = X.copy()
            X['hour'] = (X['hour'] + 1) % 24
            X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
            X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        
        return np.array(predictions)
    
    def save(self, path):
        """Sauvegarde le modle"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'feature_cols': self.feature_cols
        }, path)
        print(f" Modle sauvegard: {path}")
    
    def load(self, path):
        """Charge le modle"""
        data = joblib.load(path)
        self.xgb_model = data['xgb_model']
        self.lgb_model = data['lgb_model']
        self.feature_cols = data['feature_cols']
        print(f" Modle charg: {path}")


if __name__ == "__main__":
    # Test rapide
    from pathlib import Path
    import sys
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    # Charger donnes
    data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    stations = ["OUG_ZOG", "OUG_PATT", "BB_DIMA", "BB_KOURB", "BB_ACCART"]
    
    for station in stations[:1]:  # Test sur 1 station
        csv_file = data_path / f"{station}_historical.csv"
        if csv_file.exists():
            print(f"\n{'='*60}")
            print(f" Station: {station}")
            print('='*60)
            
            df = pd.read_csv(csv_file, parse_dates=['timestamp'])
            
            model = EnsembleForecastModel()
            metrics = model.train(df)
            
            # Sauvegarde
            model_path = Path(__file__).parent / f"{station}_ensemble.joblib"
            model.save(model_path)
            
            # Test prdiction
            predictions = model.predict(df, horizon_hours=24)
            print(f"\n Prdictions 24h: Min={predictions.min():.1f}, Max={predictions.max():.1f}, Moy={predictions.mean():.1f} m/h")
