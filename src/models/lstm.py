import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.models.base_model import BaseModel


class LSTMModel(BaseModel):
    """
    LSTM model for time series forecasting.
    
    Args:
        input_dim: Number of input features
        hidden_size: Hidden size of LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        forecast_horizon: Number of time steps to predict
        window_size: Number of time steps to look back
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to run the model on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = None
    ):
        self.input_dim = input_dim
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.forecast_horizon = forecast_horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._build_model()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _build_model(self):
        """Build the LSTM model."""
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=False  # [seq_len, batch, features]
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_size, self.forecast_horizon)
        
        # Move model to device
        self.lstm.to(self.device)
        self.output_layer.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            list(self.lstm.parameters()) +
            list(self.output_layer.parameters()),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()
    
    def _prepare_data(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.DataFrame] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert DataFrame to PyTorch tensors and reshape for sequence."""
        # Reshape X to [batch_size, window_size, input_dim]
        X_values = X.values.reshape(-1, self.window_size, self.input_dim)
        X_tensor = torch.FloatTensor(X_values).to(self.device)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y.values).to(self.device)
            return X_tensor, y_tensor
        
        return X_tensor, None
    
    def _create_batches(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.utils.data.DataLoader:
        """Create batches for training."""
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        return dataloader
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None
    ) -> "LSTMModel":
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Prepare data
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
        
        # Create batches
        train_loader = self._create_batches(X_train_tensor, y_train_tensor)
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.lstm.train()
            self.output_layer.train()
            
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # batch_X shape: [batch_size, window_size, input_dim]
                # Transpose to [window_size, batch_size, input_dim]
                batch_X = batch_X.transpose(0, 1)
                
                # LSTM forward pass
                lstm_output, (hidden, cell) = self.lstm(batch_X)
                # lstm_output shape: [window_size, batch_size, hidden_size]
                
                # Use the last output for prediction
                last_output = lstm_output[-1, :, :]  # [batch_size, hidden_size]
                
                # Predict future values
                predictions = self.output_layer(last_output)  # [batch_size, forecast_horizon]
                
                # Compute loss
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val_tensor, y_val_tensor)
                self.history['val_loss'].append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], "
                          f"Train Loss: {avg_train_loss:.4f}")
        
        print("Training completed!")
        return self
    
    def _evaluate(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        """Evaluate the model on validation data."""
        self.lstm.eval()
        self.output_layer.eval()
        
        with torch.no_grad():
            # Transpose to [window_size, batch_size, input_dim]
            X_tensor = X_tensor.transpose(0, 1)
            
            # Forward pass
            lstm_output, _ = self.lstm(X_tensor)
            last_output = lstm_output[-1, :, :]
            predictions = self.output_layer(last_output)
            
            val_loss = self.criterion(predictions, y_tensor).item()
        
        return val_loss
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            predictions: Numpy array of shape (n_samples, forecast_horizon)
        """
        self.lstm.eval()
        self.output_layer.eval()
        
        X_tensor, _ = self._prepare_data(X)
        
        with torch.no_grad():
            # Transpose to [window_size, batch_size, input_dim]
            X_tensor = X_tensor.transpose(0, 1)
            
            # Forward pass
            lstm_output, _ = self.lstm(X_tensor)
            last_output = lstm_output[-1, :, :]
            predictions = self.output_layer(last_output)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        torch.save({
            'lstm_state_dict': self.lstm.state_dict(),
            'output_layer_state_dict': self.output_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.input_dim,
                'window_size': self.window_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'forecast_horizon': self.forecast_horizon,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> "LSTMModel":
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.window_size = config['window_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.forecast_horizon = config['forecast_horizon']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        # Rebuild model
        self._build_model()
        
        # Load state dicts
        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.output_layer.load_state_dict(checkpoint['output_layer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")
        return self