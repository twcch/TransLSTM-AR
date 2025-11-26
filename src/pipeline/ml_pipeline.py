import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split

from src.data.data_loader import DataLoader
from src.preprocessing.preprocessor import (
    PreprocessingPipeline,
    MissingValueHandler,
    MinMaxScaler,
    DropColumnsPreprocessor,
    OutlierHandler
)
from src.features.feature_engineer import TimeSeriesTransformer
from src.models.base_model import BaseModel
from src.models.transformer import TransformerModel
from src.models.trans_lstm import TransLSTMModel
from utils.data import read_csv


class MLPipeline:
    """
    Machine Learning Pipeline for time series forecasting.
    
    This pipeline handles the complete workflow from data loading to model training:
    1. Data Loading
    2. Preprocessing
    3. Feature Engineering
    4. Model Training
    5. Model Saving
    
    Args:
        ticker: Stock ticker symbol
        window_size: Number of time steps to look back
        forecast_horizon: Number of time steps to predict ahead
        target_column: Name of the target column to predict
        feature_columns: List of feature column names
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        ticker: str,
        window_size: int = 30,
        forecast_horizon: int = 5,
        target_column: str = "Close",
        feature_columns: list = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.ticker = ticker
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.random_state = random_state
        
        # Pipeline components
        self.data_loader = None
        self.preprocessor = None
        self.feature_engineer = None
        self.model = None
        
        # Data
        self.raw_data = None
        self.preprocessed_data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_data(
        self,
        start: str = "2013-12-01",
        end: str = "2023-12-31",
        from_csv: bool = True,
        csv_path: str = None
    ) -> pd.DataFrame:
        """
        Load stock price data from CSV or download from API.
        
        Args:
            start: Start date for data loading (used when from_csv=False)
            end: End date for data loading (used when from_csv=False)
            from_csv: Whether to load from CSV file or download from API
            csv_path: Path to CSV file (default: data/raw/{ticker}.csv)
            
        Returns:
            raw_data: Raw stock price data
        """
        print(f"Loading data for {self.ticker}...")
        
        if from_csv:
            # Load from CSV
            if csv_path is None:
                # Clean ticker name for file path (remove dots)
                ticker_clean = self.ticker.replace('.', '')
                csv_path = f"data/raw/{ticker_clean}.csv"
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}. "
                    f"Please set from_csv=False to download data from API."
                )
            
            print(f"Loading data from {csv_path}...")
            self.raw_data = read_csv(csv_path)
            print(f"Loaded {len(self.raw_data)} records from CSV")
        else:
            # Download from API
            print(f"Downloading data from API...")
            self.data_loader = DataLoader(self.ticker, start=start, end=end)
            self.raw_data = self.data_loader.get_stock_price()
            print(f"Downloaded {len(self.raw_data)} records")
        
        return self.raw_data
    
    def preprocess_data(
        self,
        drop_columns: list = None,
        handle_missing: bool = True,
        handle_outliers: bool = False,
        scale_data: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess the data.
        
        Args:
            drop_columns: Columns to drop (default: ["Date"])
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            scale_data: Whether to scale the data
            
        Returns:
            preprocessed_data: Preprocessed data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        print("Preprocessing data...")
        
        # Build preprocessing pipeline
        steps = []
        
        # Drop columns
        if drop_columns is None:
            drop_columns = ["Date"]
        if drop_columns:
            steps.append(DropColumnsPreprocessor(columns_to_drop=drop_columns))
        
        # Handle missing values
        if handle_missing:
            steps.append(MissingValueHandler(strategy="mean"))
        
        # Handle outliers
        if handle_outliers:
            steps.append(OutlierHandler(method="iqr", threshold=1.5))
        
        # Scale data
        if scale_data:
            steps.append(MinMaxScaler(feature_range=(0, 1)))
        
        self.preprocessor = PreprocessingPipeline(steps)
        self.preprocessed_data = self.preprocessor.fit_transform(self.raw_data)
        
        print(f"Preprocessed data shape: {self.preprocessed_data.shape}")
        return self.preprocessed_data
    
    def create_sequences(self) -> tuple:
        """
        Create time series sequences using sliding windows.
        
        Returns:
            X, y: Feature and target sequences
        """
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data. Please call preprocess_data() first.")
        
        print("Creating time series sequences...")
        
        self.feature_engineer = TimeSeriesTransformer(
            window_size=self.window_size,
            forecast_horizon=self.forecast_horizon,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            stride=1
        )
        
        X, y = self.feature_engineer.fit_transform(self.preprocessed_data)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> tuple:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature sequences
            y: Target sequences
            
        Returns:
            X_train, X_val, y_train, y_val: Split datasets
        """
        print(f"Splitting data (test_size={self.test_size})...")
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            shuffle=False,
            random_state=self.random_state
        )
        
        print(f"Train size: {len(self.X_train)}, Val size: {len(self.X_val)}")
        
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def train_model(
        self,
        model_type: str = "transformer",
        model_params: Dict[str, Any] = None
    ) -> BaseModel:
        """
        Train the model.
        
        Args:
            model_type: Type of model to train ("transformer" or "trans_lstm")
            model_params: Model hyperparameters
            
        Returns:
            model: Trained model
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data. Please call split_data() first.")
        
        print(f"Training {model_type} model...")
        
        # Default parameters
        default_params = {
            "d_model": 64,
            "nhead": 8,
            "num_encoder_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
        
        if model_params is not None:
            default_params.update(model_params)
        
        # Calculate input_dim
        if self.feature_columns is None:
            n_features = len(self.preprocessed_data.columns)
            if self.target_column in self.preprocessed_data.columns:
                n_features -= 1
        else:
            n_features = len(self.feature_columns)
        
        # Create model
        if model_type.lower() == "transformer":
            self.model = TransformerModel(
                input_dim=self.X_train.shape[1],
                forecast_horizon=self.forecast_horizon,
                **default_params
            )
        elif model_type.lower() == "trans_lstm":
            self.model = TransLSTMModel(
                input_dim=n_features,
                window_size=self.window_size,
                forecast_horizon=self.forecast_horizon,
                lstm_hidden_size=default_params.get("lstm_hidden_size", 128),
                lstm_num_layers=default_params.get("lstm_num_layers", 2),
                **{k: v for k, v in default_params.items() 
                   if k not in ["lstm_hidden_size", "lstm_num_layers"]}
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val
        )
        
        print("Training completed!")
        return self.model
    
    def save_model(
        self,
        model_name: str = None,
        output_dir: str = "output/models"
    ) -> str:
        """
        Save the trained model.
        
        Args:
            model_name: Name of the model file (auto-generated if None)
            output_dir: Directory to save the model
            
        Returns:
            save_path: Path where the model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Please call train_model() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate model name if not provided
        if model_name is None:
            model_type = self.model.__class__.__name__.lower()
            model_name = f"{self.ticker}_{model_type}_w{self.window_size}_h{self.forecast_horizon}.pth"
        
        save_path = os.path.join(output_dir, model_name)
        
        print(f"Saving model to {save_path}...")
        self.model.save(save_path)
        
        return save_path
    
    def run_training_pipeline(
        self,
        start: str = "2013-12-01",
        end: str = "2023-12-31",
        from_csv: bool = True,
        csv_path: str = None,
        model_type: str = "transformer",
        model_params: Dict[str, Any] = None,
        drop_columns: list = None,
        handle_missing: bool = True,
        handle_outliers: bool = False,
        scale_data: bool = True,
        save_model: bool = True,
        model_name: str = None,
        output_dir: str = "output/models"
    ) -> BaseModel:
        """
        Run the complete training pipeline.
        
        Args:
            start: Start date for data loading
            end: End date for data loading
            from_csv: Whether to load from CSV file or download from API
            csv_path: Path to CSV file
            model_type: Type of model to train
            model_params: Model hyperparameters
            drop_columns: Columns to drop in preprocessing
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            scale_data: Whether to scale the data
            save_model: Whether to save the trained model
            model_name: Name of the model file
            output_dir: Directory to save the model
            
        Returns:
            model: Trained model
        """
        print("=" * 80)
        print(f"Starting ML Training Pipeline for {self.ticker}")
        print("=" * 80)
        
        # 1. Load data
        self.load_data(start=start, end=end, from_csv=from_csv, csv_path=csv_path)
        
        # 2. Preprocess data
        self.preprocess_data(
            drop_columns=drop_columns,
            handle_missing=handle_missing,
            handle_outliers=handle_outliers,
            scale_data=scale_data
        )
        
        # 3. Create sequences
        X, y = self.create_sequences()
        
        # 4. Split data
        self.split_data(X, y)
        
        # 5. Train model
        self.train_model(model_type=model_type, model_params=model_params)
        
        # 6. Save model
        if save_model:
            self.save_model(model_name=model_name, output_dir=output_dir)
        
        print("=" * 80)
        print("Training Pipeline Completed!")
        print("=" * 80)
        
        return self.model

    def load_model_for_inference(
        self,
        model_path: str,
        model_type: str = "transformer",
        model_params: dict = None
    ) -> BaseModel:
        """
        Load a trained model for inference.
        
        Args:
            model_path: Path to the saved model
            model_type: Type of model ("transformer" or "trans_lstm")
            model_params: Model parameters (optional, will use defaults if not provided)
            
        Returns:
            model: Loaded model
        """
        from src.models.transformer import TransformerModel
        from src.models.trans_lstm import TransLSTMModel
        
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate input_dim
        if self.feature_columns is None:
            n_features = 5  # Default: Open, High, Low, Close, Volume
        else:
            n_features = len(self.feature_columns)
        
        # Default parameters if not provided
        if model_params is None:
            model_params = {
                "d_model": 64,
                "nhead": 8,
                "num_encoder_layers": 3,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "batch_size": 32,
                "lstm_hidden_size": 128,
                "lstm_num_layers": 2
            }
        
        # Create model instance with correct architecture
        if model_type.lower() == "transformer":
            self.model = TransformerModel(
                input_dim=n_features,
                forecast_horizon=self.forecast_horizon,
                d_model=model_params.get("d_model", 64),
                nhead=model_params.get("nhead", 8),
                num_encoder_layers=model_params.get("num_encoder_layers", 3),
                dim_feedforward=model_params.get("dim_feedforward", 256),
                dropout=model_params.get("dropout", 0.1),
                learning_rate=model_params.get("learning_rate", 0.001),
                batch_size=model_params.get("batch_size", 32),
                epochs=model_params.get("epochs", 100)
            )
        elif model_type.lower() == "trans_lstm":
            self.model = TransLSTMModel(
                input_dim=n_features,
                window_size=self.window_size,
                forecast_horizon=self.forecast_horizon,
                d_model=model_params.get("d_model", 64),
                nhead=model_params.get("nhead", 8),
                num_encoder_layers=model_params.get("num_encoder_layers", 3),
                dim_feedforward=model_params.get("dim_feedforward", 256),
                dropout=model_params.get("dropout", 0.1),
                lstm_hidden_size=model_params.get("lstm_hidden_size", 128),
                lstm_num_layers=model_params.get("lstm_num_layers", 2),
                learning_rate=model_params.get("learning_rate", 0.001),
                batch_size=model_params.get("batch_size", 32),
                epochs=model_params.get("epochs", 100)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model state
        self.model.load(model_path)
        
        print("Model loaded successfully!")
        return self.model

    def run_inference_pipeline(
        self,
        model_path: str,
        model_type: str = "transformer",
        model_params: dict = None,
        start: str = "2013-12-01",
        end: str = "2023-12-31",
        from_csv: bool = True,
        csv_path: str = None,
        drop_columns: list = None,
        handle_missing: bool = True,
        scale_data: bool = True,
        save_results: bool = True,
        output_dir: str = "output/results"
    ) -> tuple:
        """
        Run the complete inference pipeline using a pre-trained model.
        
        Args:
            model_path: Path to the trained model file
            model_type: Type of model ('transformer' or 'trans_lstm')
            model_params: Model parameters (should match training configuration)
            start: Start date for data loading
            end: End date for data loading
            from_csv: Whether to load from CSV file
            csv_path: Path to CSV file
            drop_columns: Columns to drop in preprocessing
            handle_missing: Whether to handle missing values
            scale_data: Whether to scale the data
            save_results: Whether to save results to CSV
            output_dir: Directory to save results
            
        Returns:
            predictions: Array of predictions
            actual: Array of actual values
            dates: Array of dates
        """
        print("=" * 80)
        print(f"Starting Inference Pipeline for {self.ticker}")
        print(f"Using pre-trained model: {model_path}")
        print("=" * 80)
        
        # 1. Load data
        raw_data = self.load_data(start=start, end=end, from_csv=from_csv, csv_path=csv_path)
        
        # Store dates before preprocessing
        dates = raw_data['Date'].values if 'Date' in raw_data.columns else None
        
        # 2. Preprocess data (fit and transform for inference data)
        self.preprocess_data(
            drop_columns=drop_columns,
            handle_missing=handle_missing,
            handle_outliers=False,
            scale_data=scale_data
        )
        
        # 3. Create sequences
        X, y = self.create_sequences()
        
        print(f"Created {len(X)} sequences for inference")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # 4. Load pre-trained model with correct parameters
        self.load_model_for_inference(
            model_path=model_path,
            model_type=model_type,
            model_params=model_params
        )
        
        # 5. Make predictions
        print("Making predictions...")
        predictions = self.model.predict(X)
        
        # Get the first forecast for each sequence (t+1 prediction)
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = predictions[:, 0]
        else:
            predictions = predictions.flatten()
        
        # Get actual values
        if len(y.shape) > 1 and y.shape[1] > 1:
            actual = y.iloc[:, 0].values
        else:
            actual = y.values.flatten()
        
        # 6. Inverse transform if scaler exists
        if scale_data and hasattr(self, 'preprocessor') and self.preprocessor is not None:
            from src.preprocessing.preprocessor import MinMaxScaler
            
            for step in self.preprocessor.steps:
                if isinstance(step, MinMaxScaler):
                    target_idx = step.numeric_columns.index(self.target_column)
                    n_features = len(step.numeric_columns)
                    
                    # Predictions
                    pred_full = np.zeros((len(predictions), n_features))
                    pred_full[:, target_idx] = predictions
                    pred_inverse = step.scaler.inverse_transform(pred_full)
                    predictions = pred_inverse[:, target_idx]
                    
                    # Actual values
                    actual_full = np.zeros((len(actual), n_features))
                    actual_full[:, target_idx] = actual
                    actual_inverse = step.scaler.inverse_transform(actual_full)
                    actual = actual_inverse[:, target_idx]
                    break
        
        # Get corresponding dates
        if dates is not None:
            inference_dates = dates[self.window_size:self.window_size + len(predictions)]
        else:
            inference_dates = np.arange(len(predictions))
        
        print(f"Inference complete. Generated {len(predictions)} predictions")
        
        # 7. Save results if requested
        if save_results:
            self._save_inference_results(
                predictions=predictions,
                actual=actual,
                dates=inference_dates,
                output_dir=output_dir
            )
        
        print("=" * 80)
        print("Inference Pipeline Completed!")
        print("=" * 80)
        
        return predictions, actual, inference_dates
    
    def _save_inference_results(
        self,
        predictions: np.ndarray,
        actual: np.ndarray,
        dates: np.ndarray,
        output_dir: str = "output/results"
    ) -> str:
        """
        Save inference results to CSV.
        
        Args:
            predictions: Predicted values
            actual: Actual values
            dates: Dates corresponding to predictions
            output_dir: Directory to save results
            
        Returns:
            save_path: Path where results were saved
        """
        from datetime import datetime
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual': actual,
            'Predicted': predictions,
            'Error': actual - predictions,
            'Abs_Error': np.abs(actual - predictions),
            'Percentage_Error': np.abs((actual - predictions) / actual) * 100
        })
        
        # Calculate metrics
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        # Print metrics
        print(f"\nMetrics for {self.ticker}:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker_clean = self.ticker.replace('.', '')
        model_type = self.model.__class__.__name__.lower().replace('model', '')
        
        results_path = os.path.join(
            output_dir,
            f"{ticker_clean}_{model_type}_predictions_{timestamp}.csv"
        )
        results_df.to_csv(results_path, index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE', 'MSE'],
            'Value': [rmse, mae, mape, mse]
        })
        
        metrics_path = os.path.join(
            output_dir,
            f"{ticker_clean}_{model_type}_metrics_{timestamp}.csv"
        )
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"\nResults saved to: {results_path}")
        print(f"Metrics saved to: {metrics_path}")
        
        return results_path
