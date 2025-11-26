import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.models.base_model import BaseModel


class MLPLSTMModel(BaseModel):
    """
    MLP-LSTM hybrid model for time series forecasting.
    
    The model uses MLP (Multi-Layer Perceptron) for feature extraction
    and LSTM for autoregressive time series prediction.
    
    Architecture:
    1. Input -> MLP layers -> Feature extraction
    2. MLP output -> LSTM -> Temporal modeling
    3. LSTM output -> Autoregressive prediction -> Forecast
    
    Args:
        input_dim: Number of input features
        window_size: Number of time steps to look back
        mlp_hidden_sizes: List of hidden layer sizes for MLP (e.g., [128, 64])
        lstm_hidden_size: Hidden size of LSTM
        lstm_num_layers: Number of LSTM layers
        dropout: Dropout rate
        forecast_horizon: Number of time steps to predict
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        device: Device to run the model on ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        mlp_hidden_sizes: list = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        forecast_horizon: int = 5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = None
    ):
        self.input_dim = input_dim
        self.window_size = window_size
        self.mlp_hidden_sizes = mlp_hidden_sizes if mlp_hidden_sizes is not None else [128, 64]
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
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
        """Build the MLP-LSTM model."""
        # MLP for feature extraction
        mlp_layers = []
        prev_size = self.input_dim
        
        for hidden_size in self.mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # LSTM for temporal modeling
        # Input size is the last MLP hidden size
        self.lstm = nn.LSTM(
            input_size=self.mlp_hidden_sizes[-1],
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.dropout if self.lstm_num_layers > 1 else 0,
            batch_first=False  # [seq_len, batch, features]
        )
        
        # Output layer for prediction
        self.output_layer = nn.Linear(self.lstm_hidden_size, 1)
        
        # Move model to device
        self.mlp.to(self.device)
        self.lstm.to(self.device)
        self.output_layer.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            list(self.mlp.parameters()) +
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
    ) -> "MLPLSTMModel":
        """
        Train the MLP-LSTM model.
        
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
            self.mlp.train()
            self.lstm.train()
            self.output_layer.train()
            
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # batch_X shape: [batch_size, window_size, input_dim]
                # Transpose to [window_size, batch_size, input_dim]
                batch_X = batch_X.transpose(0, 1)
                
                # MLP feature extraction for each time step
                # Process each time step through MLP
                mlp_outputs = []
                for t in range(batch_X.size(0)):  # iterate over time steps
                    time_step = batch_X[t, :, :]  # [batch_size, input_dim]
                    mlp_out = self.mlp(time_step)  # [batch_size, mlp_hidden_sizes[-1]]
                    mlp_outputs.append(mlp_out)
                
                # Stack MLP outputs
                mlp_sequence = torch.stack(mlp_outputs, dim=0)  # [window_size, batch_size, mlp_hidden]
                
                # LSTM processes the MLP feature sequence
                lstm_output, hidden = self.lstm(mlp_sequence)  # [window_size, batch_size, lstm_hidden]
                
                # Autoregressive prediction for future steps
                predictions = []
                # Use the last LSTM hidden state for future predictions
                current_hidden = hidden
                # Use the last MLP output as initial input
                current_input = mlp_sequence[-1:, :, :]  # [1, batch_size, mlp_hidden]
                
                for t in range(self.forecast_horizon):
                    # LSTM step with current input
                    lstm_out, current_hidden = self.lstm(current_input, current_hidden)  # [1, batch_size, lstm_hidden]
                    
                    # Predict next value
                    pred = self.output_layer(lstm_out.squeeze(0))  # [batch_size, 1]
                    predictions.append(pred)
                    
                    # Keep using the same input for next step
                    # (In practice, you might want to feed back the prediction)
                
                # Stack predictions
                predictions = torch.cat(predictions, dim=1)  # [batch_size, forecast_horizon]
                
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
        
        return self
    
    def _evaluate(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        """Evaluate the model on validation data."""
        self.mlp.eval()
        self.lstm.eval()
        self.output_layer.eval()
        
        with torch.no_grad():
            # Transpose to [window_size, batch_size, input_dim]
            X_tensor = X_tensor.transpose(0, 1)
            
            # MLP feature extraction
            mlp_outputs = []
            for t in range(X_tensor.size(0)):
                time_step = X_tensor[t, :, :]
                mlp_out = self.mlp(time_step)
                mlp_outputs.append(mlp_out)
            
            mlp_sequence = torch.stack(mlp_outputs, dim=0)
            
            # LSTM processes the MLP feature sequence
            lstm_output, hidden = self.lstm(mlp_sequence)
            
            # Autoregressive prediction
            predictions = []
            current_hidden = hidden
            current_input = mlp_sequence[-1:, :, :]
            
            for t in range(self.forecast_horizon):
                lstm_out, current_hidden = self.lstm(current_input, current_hidden)
                pred = self.output_layer(lstm_out.squeeze(0))
                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=1)
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
        self.mlp.eval()
        self.lstm.eval()
        self.output_layer.eval()
        
        X_tensor, _ = self._prepare_data(X)
        
        with torch.no_grad():
            # Transpose to [window_size, batch_size, input_dim]
            X_tensor = X_tensor.transpose(0, 1)
            
            # MLP feature extraction
            mlp_outputs = []
            for t in range(X_tensor.size(0)):
                time_step = X_tensor[t, :, :]
                mlp_out = self.mlp(time_step)
                mlp_outputs.append(mlp_out)
            
            mlp_sequence = torch.stack(mlp_outputs, dim=0)
            
            # LSTM processes the MLP feature sequence
            lstm_output, hidden = self.lstm(mlp_sequence)
            
            # Autoregressive prediction for future steps
            predictions = []
            current_hidden = hidden
            current_input = mlp_sequence[-1:, :, :]
            
            for t in range(self.forecast_horizon):
                lstm_out, current_hidden = self.lstm(current_input, current_hidden)
                pred = self.output_layer(lstm_out.squeeze(0))
                predictions.append(pred)
            
            # Stack predictions
            predictions = torch.cat(predictions, dim=1)  # [batch_size, forecast_horizon]
        
        return predictions.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        torch.save({
            'mlp_state_dict': self.mlp.state_dict(),
            'lstm_state_dict': self.lstm.state_dict(),
            'output_layer_state_dict': self.output_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.input_dim,
                'window_size': self.window_size,
                'mlp_hidden_sizes': self.mlp_hidden_sizes,
                'lstm_hidden_size': self.lstm_hidden_size,
                'lstm_num_layers': self.lstm_num_layers,
                'dropout': self.dropout,
                'forecast_horizon': self.forecast_horizon,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> "MLPLSTMModel":
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.window_size = config['window_size']
        self.mlp_hidden_sizes = config['mlp_hidden_sizes']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.lstm_num_layers = config['lstm_num_layers']
        self.dropout = config['dropout']
        self.forecast_horizon = config['forecast_horizon']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        # Rebuild model
        self._build_model()
        
        # Load state dicts
        self.mlp.load_state_dict(checkpoint['mlp_state_dict'])
        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.output_layer.load_state_dict(checkpoint['output_layer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")
        return self