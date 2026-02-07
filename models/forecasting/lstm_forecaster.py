"""
Modle LSTM pour prdiction de la demande en eau et nergie
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))


class LSTMForecaster(nn.Module):
    """
    LSTM simplifié pour prdiction rapide sur CPU
    Architecture optimisée pour vitesse
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 1,
                 forecast_horizon: int = 24,
                 dropout: float = 0.3):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # LSTM unidirectionnel (plus rapide sur CPU)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Output layers simplifiés (2 couches au lieu de 3)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, forecast_horizon * output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, input_size)
        Returns:
            predictions: (batch_size, forecast_horizon, output_size)
        """
        # LSTM unidirectionnel
        lstm_out, _ = self.lstm(x)
        
        # Prendre dernière sortie
        last_output = lstm_out[:, -1, :]
        
        # Dense layers (2 couches)
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Reshape pour multi-horizon
        out = out.view(-1, self.forecast_horizon, 1)
        
        return out


class EnergyDemandForecaster:
    """
    Wrapper pour entranement et prdiction
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        
    def build_model(self, input_size: int):
        """Construit le modle"""
        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=self.config.get('hidden_size', 64),
            num_layers=self.config.get('num_layers', 2),
            output_size=1,
            forecast_horizon=self.config.get('forecast_horizon', 24),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        
        print(f" Modle LSTM construit sur {self.device}")
        print(f"  - Paramtres: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train(self, train_loader, val_loader, epochs=50):
        """
        Entraînement LSTM optimisé CPU avec:
        - Early stopping fiable
        - Validation rapide
        - Architecture simplifiée
        """
        
        device = self.device
        model = self.model
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # ==============================
        # EARLY STOPPING - CORRECT
        # ==============================
        best_val_loss = float("inf")
        patience = 7
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        print(f"🔄 Entraînement LSTM ({device})")
        
        for epoch in range(epochs):
            # ==============================
            # TRAIN
            # ==============================
            model.train()
            train_loss_epoch = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()
                
                optimizer.zero_grad(set_to_none=True)
                
                output = model(X_batch)
                loss = criterion(output, y_batch)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss_epoch += loss.item()
            
            train_loss_epoch /= len(train_loader)
            train_losses.append(train_loss_epoch)
            
            # ==============================
            # VALIDATION (NO GRAD)
            # ==============================
            model.eval()
            val_loss_epoch = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device).float()
                    y_batch = y_batch.to(device).float()
                    
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    
                    val_loss_epoch += loss.item()
            
            val_loss_epoch /= len(val_loader)
            val_losses.append(val_loss_epoch)
            
            # ==============================
            # LR SCHEDULER
            # ==============================
            scheduler.step(val_loss_epoch)
            
            # ==============================
            # EARLY STOPPING LOGIC
            # ==============================
            if val_loss_epoch < best_val_loss - 1e-6:
                best_val_loss = val_loss_epoch
                patience_counter = 0
                
                # Sauvegarde du meilleur modèle
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            # ==============================
            # LOG
            # ==============================
            print(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train={train_loss_epoch:.4f} | "
                f"Val={val_loss_epoch:.4f} | "
                f"Best={best_val_loss:.4f} | "
                f"Pat={patience_counter}/{patience}"
            )
            
            # ==============================
            # STOP
            # ==============================
            if patience_counter >= patience:
                print("🛑 Early stopping déclenché")
                break
        
        # ==============================
        # LOAD BEST MODEL
        # ==============================
        self.load_checkpoint('best_model.pth')
        
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prdiction"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(0)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def save_checkpoint(self, filename: str):
        """Sauvegarde le modle"""
        models_dir = Path(__file__).parent
        models_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, models_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Charge le modle"""
        models_dir = Path(__file__).parent
        checkpoint = torch.load(models_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule mtriques de performance"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


if __name__ == "__main__":
    print(" Test du modle LSTM Forecaster")
    
    # Configuration test
    config = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'forecast_horizon': 24
    }
    
    # Crer donnes dummy
    batch_size = 32
    seq_length = 168  # 1 semaine
    input_size = 10
    
    X_dummy = torch.randn(batch_size, seq_length, input_size)
    
    # Tester le modle
    forecaster = EnergyDemandForecaster(config)
    forecaster.build_model(input_size)
    
    # Forward pass
    with torch.no_grad():
        output = forecaster.model(X_dummy)
    
    print(f"\n Test forward pass russi!")
    print(f"  - Input shape: {X_dummy.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Expected: (batch={batch_size}, horizon=24, features=1)")

