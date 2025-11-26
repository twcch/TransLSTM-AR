import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from src.models.base_model import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransLSTMModel(BaseModel):
    """
    Transformer-LSTM hybrid model for time series forecasting.
    
    The model uses Transformer encoder to capture long-term dependencies
    and LSTM decoder for autoregressive prediction.
    
    Args:
        input_dim: Number of input features
        d_model: Dimension of the model (embedding dimension)
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        lstm_hidden_size: Hidden size of LSTM
        lstm_num_layers: Number of LSTM layers
        dim_feedforward: Dimension of feedforward network in transformer
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
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        forecast_horizon: int = 5,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = None
    ):
        self.input_dim = input_dim
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dim_feedforward = dim_feedforward
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
        """Build the TransLSTM model."""
        # Input embedding layer
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
        # LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.dropout if self.lstm_num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.lstm_hidden_size, 1)
        
        # Move model to device
        self.input_embedding.to(self.device)
        self.pos_encoder.to(self.device)
        self.transformer_encoder.to(self.device)
        self.lstm.to(self.device)
        self.output_layer.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(
            list(self.input_embedding.parameters()) +
            list(self.transformer_encoder.parameters()) +
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
    ) -> "TransLSTMModel":
        """
        Train the TransLSTM model.
        
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
            self.input_embedding.train()
            self.transformer_encoder.train()
            self.lstm.train()
            self.output_layer.train()
            
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # batch_X shape: [batch_size, window_size, input_dim]
                # Transpose to [window_size, batch_size, input_dim]
                batch_X = batch_X.transpose(0, 1)
                
                # Input embedding
                embedded = self.input_embedding(batch_X)  # [window_size, batch_size, d_model]
                
                # Positional encoding
                embedded = self.pos_encoder(embedded)
                
                # Transformer encoder
                encoder_output = self.transformer_encoder(embedded)  # [window_size, batch_size, d_model]
                
                # LSTM decoder with autoregressive prediction
                predictions = []
                hidden = None
                
                # Use last encoder output as initial input to LSTM
                lstm_input = encoder_output[-1:, :, :]  # [1, batch_size, d_model]
                
                for t in range(self.forecast_horizon):
                    # LSTM step
                    lstm_output, hidden = self.lstm(lstm_input, hidden)  # lstm_output: [1, batch_size, lstm_hidden]
                    
                    # Predict next value
                    pred = self.output_layer(lstm_output.squeeze(0))  # [batch_size, 1]
                    predictions.append(pred)
                    
                    # Use prediction as next input (teacher forcing could be added here)
                    # For now, we continue with the lstm output
                    lstm_input = lstm_output
                
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
        self.input_embedding.eval()
        self.transformer_encoder.eval()
        self.lstm.eval()
        self.output_layer.eval()
        
        with torch.no_grad():
            # Transpose to [window_size, batch_size, input_dim]
            X_tensor = X_tensor.transpose(0, 1)
            
            # Forward pass
            embedded = self.input_embedding(X_tensor)
            embedded = self.pos_encoder(embedded)
            encoder_output = self.transformer_encoder(embedded)
            
            predictions = []
            hidden = None
            lstm_input = encoder_output[-1:, :, :]
            
            for t in range(self.forecast_horizon):
                lstm_output, hidden = self.lstm(lstm_input, hidden)
                pred = self.output_layer(lstm_output.squeeze(0))
                predictions.append(pred)
                lstm_input = lstm_output
            
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
        self.input_embedding.eval()
        self.transformer_encoder.eval()
        self.lstm.eval()
        self.output_layer.eval()
        
        X_tensor, _ = self._prepare_data(X)
        
        with torch.no_grad():
            # Transpose to [window_size, batch_size, input_dim]
            X_tensor = X_tensor.transpose(0, 1)
            
            # Forward pass
            embedded = self.input_embedding(X_tensor)
            embedded = self.pos_encoder(embedded)
            encoder_output = self.transformer_encoder(embedded)
            
            predictions = []
            hidden = None
            lstm_input = encoder_output[-1:, :, :]
            
            for t in range(self.forecast_horizon):
                lstm_output, hidden = self.lstm(lstm_input, hidden)
                pred = self.output_layer(lstm_output.squeeze(0))
                predictions.append(pred)
                lstm_input = lstm_output
            
            predictions = torch.cat(predictions, dim=1)
        
        return predictions.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        torch.save({
            'input_embedding_state_dict': self.input_embedding.state_dict(),
            'transformer_encoder_state_dict': self.transformer_encoder.state_dict(),
            'lstm_state_dict': self.lstm.state_dict(),
            'output_layer_state_dict': self.output_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.input_dim,
                'window_size': self.window_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'lstm_hidden_size': self.lstm_hidden_size,
                'lstm_num_layers': self.lstm_num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'forecast_horizon': self.forecast_horizon,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> "TransLSTMModel":
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        config = checkpoint['config']
        self.input_dim = config['input_dim']
        self.window_size = config['window_size']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.lstm_hidden_size = config['lstm_hidden_size']
        self.lstm_num_layers = config['lstm_num_layers']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.forecast_horizon = config['forecast_horizon']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        # Rebuild model
        self._build_model()
        
        # Load state dicts
        self.input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
        self.transformer_encoder.load_state_dict(checkpoint['transformer_encoder_state_dict'])
        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.output_layer.load_state_dict(checkpoint['output_layer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")
        return self