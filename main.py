from src.data.data_loader import DataLoader
from src.pipeline.ml_pipeline import MLPipeline
from utils.data import read_csv
import pandas as pd
import os


def load_data(tickers: list[str]):
    """Download and save stock data."""
    for ticker in tickers:
        print(f"Loading data for {ticker}...")
        data_loader = DataLoader(ticker)
        data = data_loader.get_stock_price()
        data_loader.save_to_csv(f"data/raw/{ticker.replace('.', '')}.csv")
        print(f"Data for {ticker} saved to CSV.")


def combine_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Combine multiple stock data into a single DataFrame.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        
    Returns:
        Combined DataFrame with all stocks data
    """
    combined_data = []
    
    for ticker in tickers:
        print(f"Loading data for {ticker}...")
        # Load from CSV
        csv_path = f"data/raw/{ticker.replace('.', '')}.csv"
        data = read_csv(csv_path)
        
        # Filter by date range
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[(data['Date'] >= start) & (data['Date'] <= end)]
        
        # Add ticker column to identify the stock
        data['Ticker'] = ticker
        
        combined_data.append(data)
    
    # Concatenate all data
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\nCombined data shape: {combined_df.shape}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"Stocks included: {combined_df['Ticker'].unique()}")
    
    return combined_df

def train_model_multi_stock(
    tickers: list[str],
    model_type: str = "transformer",
    window_size: int = 30,
    forecast_horizon: int = 5,
    from_csv: bool = True,
    start: str = "2013-12-01",
    end: str = "2023-12-31"
) -> tuple:
    """Train a model using multiple stocks' data."""
    print(f"\n{'=' * 80}")
    print(f"Training {model_type} model with multiple stocks")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Window size: {window_size}, Forecast horizon: {forecast_horizon}")
    print(f"Data source: {'CSV' if from_csv else 'API'}")
    print(f"{'=' * 80}\n")
    
    # Combine data from multiple stocks
    combined_data = combine_stock_data(tickers, start, end)
    
    # Save combined data
    combined_csv_path = "data/raw/combined_stocks.csv"
    combined_data.to_csv(combined_csv_path, index=False)
    print(f"Combined data saved to {combined_csv_path}")
    
    # Create pipeline with a generic ticker name
    pipeline = MLPipeline(
        ticker="MultiStock",
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        target_column="Close",
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        test_size=0.2,
        random_state=42
    )
    
    # Model hyperparameters
    model_params = {
        "d_model": 64,
        "nhead": 8,
        "num_encoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
    
    # Add model-specific parameters
    if model_type == "trans_lstm":
        model_params.update({
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2
        })
    elif model_type == "lstm":
        model_params.update({
            "hidden_size": 128,
            "num_layers": 2
        })
    
    # Generate model name - DON'T specify custom name, let save_model use default format
    # This will create: MultiStock_lstmmodel_w30_h5.pth, etc.
    
    # Run training pipeline with combined data
    model = pipeline.run_training_pipeline(
        start=start,
        end=end,
        from_csv=True,
        csv_path=combined_csv_path,
        model_type=model_type,
        model_params=model_params,
        drop_columns=["Date", "Ticker"],
        handle_missing=True,
        handle_outliers=False,
        scale_data=True,
        save_model=True,
        model_name=None,  # Use default naming
        output_dir="output/models"
    )
    
    return model, pipeline

def inference_single_stock(
    ticker: str,
    model_path: str,
    model_type: str,
    window_size: int = 30,
    forecast_horizon: int = 5,
    start: str = "2013-12-01",
    end: str = "2023-12-31",
    output_dir: str = "output/results"
) -> tuple:
    """
    Run inference on a single stock using the trained global model.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to the trained model
        model_type: Type of model ('transformer', 'trans_lstm', or 'lstm')
        window_size: Window size used during training
        forecast_horizon: Forecast horizon used during training
        start: Start date for data
        end: End date for data
        output_dir: Directory to save results
        
    Returns:
        predictions, actual, dates: Inference results
    """
    print(f"\n{'=' * 80}")
    print(f"Running inference for {ticker}")
    print(f"Model: {model_type}")
    print(f"Model path: {model_path}")
    print(f"{'=' * 80}\n")
    
    # Create pipeline
    pipeline = MLPipeline(
        ticker=ticker,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        target_column="Close",
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        test_size=0.2,
        random_state=42
    )
    
    # Model parameters (must match training configuration)
    model_params = {
        "d_model": 64,
        "nhead": 8,
        "num_encoder_layers": 3,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
    
    # Add model-specific parameters
    if model_type == "trans_lstm":
        model_params.update({
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2
        })
    elif model_type == "lstm":
        model_params.update({
            "hidden_size": 128,
            "num_layers": 2
        })
    
    # Run inference pipeline
    predictions, actual, dates = pipeline.run_inference_pipeline(
        model_path=model_path,
        model_type=model_type,
        model_params=model_params,  # Pass model params here
        start=start,
        end=end,
        from_csv=True,
        csv_path=f"data/raw/{ticker.replace('.', '')}.csv",
        drop_columns=["Date"],
        handle_missing=True,
        scale_data=True,
        save_results=True,
        output_dir=output_dir
    )
    
    return predictions, actual, dates

def inference_multi_stock(
    tickers: list[str],
    model_path: str,
    model_type: str = "transformer",
    window_size: int = 30,
    forecast_horizon: int = 5,
    start: str = "2013-12-01",
    end: str = "2023-12-31",
    output_dir: str = "output/results"
):
    """
    Run inference on multiple stocks using the same trained global model.
    
    Args:
        tickers: List of stock ticker symbols
        model_path: Path to the trained model
        model_type: Type of model ('transformer' or 'trans_lstm')
        window_size: Window size used during training
        forecast_horizon: Forecast horizon used during training
        start: Start date for data
        end: End date for data
        output_dir: Directory to save results
    """
    print(f"\n{'=' * 80}")
    print(f"Running inference for multiple stocks using global model")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Model type: {model_type}")
    print(f"{'=' * 80}\n")
    
    all_metrics = []
    
    for ticker in tickers:
        try:
            predictions, actual, dates = inference_single_stock(
                ticker=ticker,
                model_path=model_path,
                model_type=model_type,
                window_size=window_size,
                forecast_horizon=forecast_horizon,
                start=start,
                end=end,
                output_dir=output_dir
            )
            
            # Calculate metrics for summary
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np
            
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            all_metrics.append({
                'Ticker': ticker,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })
            
        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            continue
    
    # Save combined metrics
    if all_metrics:
        from datetime import datetime
        
        combined_metrics = pd.DataFrame(all_metrics)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_metrics_path = os.path.join(
            output_dir,
            f"all_stocks_{model_type}_metrics_{timestamp}.csv"
        )
        combined_metrics.to_csv(combined_metrics_path, index=False)
        
        print(f"\n{'=' * 80}")
        print(f"Combined metrics saved to: {combined_metrics_path}")
        print(f"{'=' * 80}\n")
        
        # Print summary
        print("\nSummary of all stocks:")
        print(combined_metrics.to_string(index=False))

def main():
    # Mode selection
    MODE = ["train", "inference"]  # Options: "load", "train", "inference"
    
    # Define tickers to use
    tickers = ["2330.TW", "AAPL"]
    
    # Model configuration
    model_type = "lstm"  # transformer, trans_lstm, or lstm
    window_size = 30
    forecast_horizon = 5
    
    # Date range
    start = "2013-12-01"
    end = "2023-12-31"
    
    # Load data
    if "load" in MODE:
        load_data(tickers)
    
    # Train model
    if "train" in MODE:
        model, pipeline = train_model_multi_stock(
            tickers=tickers,
            model_type=model_type,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            from_csv=True,
            start=start,
            end=end
        )
    
    # Run inference (completely independent from training)
    if "inference" in MODE:
        # Generate model path using the same format as save_model
        # Format: {ticker}_{model_class}model_w{window}_h{horizon}.pth
        model_class_name = model_type  # "lstm", "translstm", or "transformer"
        if model_type == "trans_lstm":
            model_class_name = "translstm"
        
        model_filename = f"MultiStock_{model_class_name}model_w{window_size}_h{forecast_horizon}.pth"
        model_path = f"output/models/{model_filename}"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"\n{'=' * 80}")
            print(f"ERROR: Model file not found at {model_path}")
            print(f"{'=' * 80}")
            print("\nPlease train the model first by:")
            print("1. Set MODE = ['train'] in main()")
            print("2. Run the script to train and save the model")
            print("3. Then set MODE = ['inference'] to run inference")
            print(f"\nExpected model file: {model_filename}")
            print(f"{'=' * 80}\n")
            return
        
        print(f"\nFound trained model at: {model_path}")
        
        # Run inference on each stock separately using the same global model
        inference_multi_stock(
            tickers=tickers,
            model_path=model_path,
            model_type=model_type,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            start=start,
            end=end,
            output_dir="output/results"
        )

if __name__ == "__main__":
    main()