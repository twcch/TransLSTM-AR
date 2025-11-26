import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from src.data.data_loader import DataLoader
from src.pipeline.ml_pipeline import MLPipeline
from utils.data import read_csv
from sklearn.preprocessing import MinMaxScaler
from src.features.feature_engineer import TimeSeriesTransformer


def load_data(tickers: list[str]):
    """Download and save stock data."""
    for ticker in tickers:
        print(f"Loading data for {ticker}...")
        data_loader = DataLoader(ticker)
        data = data_loader.get_stock_price()
        data_loader.save_to_csv(f"data/raw/{ticker.replace('.', '')}.csv")
        print(f"Data for {ticker} saved to CSV.")


# def combine_stock_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
#     """
#     Combine multiple stock data with per-stock standardization.

#     Args:
#         tickers: List of ticker symbols
#         start: Start date
#         end: End date

#     Returns:
#         Combined DataFrame with all stocks data (already scaled per stock)
#     """
#     from sklearn.preprocessing import MinMaxScaler

#     combined_data = []

#     for ticker in tickers:
#         print(f"Loading data for {ticker}...")
#         # Load from CSV
#         csv_path = f"data/raw/{ticker.replace('.', '')}.csv"
#         data = read_csv(csv_path)

#         # Filter by date range
#         data["Date"] = pd.to_datetime(data["Date"])
#         data = data[(data["Date"] >= start) & (data["Date"] <= end)].copy()

#         # Per-stock standardization
#         # Scale numeric columns (Open, High, Low, Close, Volume)
#         numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
#         scaler = MinMaxScaler(feature_range=(0, 1))

#         # Only scale if columns exist
#         existing_numeric = [col for col in numeric_columns if col in data.columns]
#         if existing_numeric:
#             data[existing_numeric] = scaler.fit_transform(data[existing_numeric])

#         # Add ticker column to identify the stock
#         data["Ticker"] = ticker

#         combined_data.append(data)
#         print(f"  - Scaled {ticker} independently (price range normalized to [0, 1])")

#     # Concatenate all data
#     combined_df = pd.concat(combined_data, ignore_index=True)

#     # Sort by date
#     combined_df = combined_df.sort_values("Date").reset_index(drop=True)

#     print(f"\nCombined data shape: {combined_df.shape}")
#     print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
#     print(f"Stocks included: {combined_df['Ticker'].unique()}")
#     print("Note: Each stock has been scaled independently before combining")

#     return combined_df

import pickle
import os

def combine_stock_sequences(
    tickers: list[str], 
    start: str, 
    end: str, 
    window_size: int = 30,
    forecast_horizon: int = 5,
    save_scalers: bool = True,
    scaler_dir: str = "output/scalers"
) -> tuple:
    """
    Combine multiple stock data by creating sequences per stock first,
    then concatenating all sequences.
    
    This preserves time series continuity for each stock.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date
        window_size: Sliding window size
        forecast_horizon: Prediction horizon
        save_scalers: Whether to save scalers for inference
        scaler_dir: Directory to save scaler objects
        
    Returns:
        X_combined: Combined feature sequences
        y_combined: Combined target sequences
    """
    all_X_sequences = []
    all_y_sequences = []
    
    # Create scaler directory if needed
    if save_scalers:
        os.makedirs(scaler_dir, exist_ok=True)
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # 1. Load data
        csv_path = f"data/raw/{ticker.replace('.', '')}.csv"
        data = read_csv(csv_path)
        
        # 2. Filter by date range
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[(data['Date'] >= start) & (data['Date'] <= end)].copy()
        
        # 3. Per-stock standardization
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        scaler = MinMaxScaler(feature_range=(0, 1))
        existing_numeric = [col for col in numeric_columns if col in data.columns]
        
        if existing_numeric:
            # ✅ Fit on training data
            data[existing_numeric] = scaler.fit_transform(data[existing_numeric])
            
            # ✅ Save scaler for inference
            if save_scalers:
                ticker_clean = ticker.replace('.', '')
                scaler_path = os.path.join(scaler_dir, f"{ticker_clean}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump({
                        'scaler': scaler,
                        'columns': existing_numeric,
                        'ticker': ticker,
                        'date_range': (start, end)
                    }, f)
                print(f"  - Scaler saved to {scaler_path}")
        
        print(f"  - Scaled {ticker} independently")
        
        # 4. Create sequences for THIS stock only
        feature_columns = existing_numeric
        target_column = 'Close'
        
        transformer = TimeSeriesTransformer(
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            target_column=target_column,
            feature_columns=feature_columns,
            stride=1
        )
        
        X_stock, y_stock = transformer.fit_transform(data)
        
        print(f"  - Created {len(X_stock)} sequences for {ticker}")
        print(f"    X shape: {X_stock.shape}, y shape: {y_stock.shape}")
        
        # 5. Add to combined list
        all_X_sequences.append(X_stock)
        all_y_sequences.append(y_stock)
    
    # 6. Concatenate all sequences
    X_combined = pd.concat(all_X_sequences, ignore_index=True)
    y_combined = pd.concat(all_y_sequences, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"Combined sequences from {len(tickers)} stocks:")
    print(f"Total X shape: {X_combined.shape}")
    print(f"Total y shape: {y_combined.shape}")
    print(f"Each sequence maintains temporal continuity within its stock")
    if save_scalers:
        print(f"Scalers saved to: {scaler_dir}/")
    print(f"{'='*80}\n")
    
    return X_combined, y_combined

def train_model_multi_stock(
    tickers: list[str],
    model_type: str = "transformer",
    window_size: int = 30,
    forecast_horizon: int = 5,
    from_csv: bool = True,
    start: str = "2013-12-01",
    end: str = "2023-12-31",
) -> tuple:
    """Train a model using multiple stocks' sequences."""
    print(f"\n{'=' * 80}")
    print(f"Training {model_type} model with multiple stocks")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Window size: {window_size}, Forecast horizon: {forecast_horizon}")
    print(f"Data source: {'CSV' if from_csv else 'API'}")
    print(f"{'=' * 80}\n")

    # ✅ 使用正確的函數：先建立序列，再合併
    X_combined, y_combined = combine_stock_sequences(
        tickers=tickers,
        start=start,
        end=end,
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )
    
    # ✅ 驗證序列完整性（添加這個檢查）
    verify_sequence_integrity(X_combined, window_size=window_size, n_features=5)
    
    # Save combined sequences (optional)
    combined_dir = "data/processed"
    os.makedirs(combined_dir, exist_ok=True)
    X_combined.to_csv(f"{combined_dir}/X_combined.csv", index=False)
    y_combined.to_csv(f"{combined_dir}/y_combined.csv", index=False)
    print(f"Combined sequences saved to {combined_dir}/")

    # Create pipeline
    pipeline = MLPipeline(
        ticker="MultiStock",
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        target_column="Close",
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        test_size=0.2,
        random_state=42,
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
        "epochs": 100,
    }

    # Add model-specific parameters
    if model_type == "trans_lstm":
        model_params.update({"lstm_hidden_size": 128, "lstm_num_layers": 2})
    elif model_type == "lstm":
        model_params.update({"hidden_size": 128, "num_layers": 2, "dropout": 0.2})
    elif model_type == "mlp_lstm":
        model_params.update({
            "mlp_hidden_sizes": [128, 64],
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2
        })

    # ✅ 關鍵：跳過 pipeline 的 create_sequences，直接使用已建立的序列
    pipeline.preprocessed_data = None  # Not needed
    
    # Split data into train/val
    from sklearn.model_selection import train_test_split
    pipeline.X_train, pipeline.X_val, pipeline.y_train, pipeline.y_val = train_test_split(
        X_combined, 
        y_combined,
        test_size=pipeline.test_size,
        shuffle=False,  # ⚠️ 重要：不打亂時間順序
        random_state=pipeline.random_state
    )
    
    print(f"Train size: {len(pipeline.X_train)}, Val size: {len(pipeline.X_val)}")
    
    # Train model
    print(f"\nTraining {model_type} model...")
    model = pipeline.train_model(
        model_type=model_type,
        model_params=model_params
    )
    
    # Save model
    print("\nSaving model...")
    model_path = pipeline.save_model(
        model_name=None,
        output_dir="output/models"
    )
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Model saved to: {model_path}")
    print(f"{'='*80}\n")

    return model, pipeline


def verify_sequence_integrity(X: pd.DataFrame, window_size: int = 30, n_features: int = 5):
    """
    驗證序列完整性 - 確保每個序列代表單一股票的連續時間
    """
    print("\n" + "="*80)
    print("🔍 Verifying Sequence Integrity")
    print("="*80)
    
    # 檢查形狀
    expected_features = window_size * n_features
    if X.shape[1] != expected_features:
        print(f"❌ ERROR: Expected {expected_features} features, got {X.shape[1]}")
        return False
    
    print(f"✅ Shape correct: {X.shape}")
    print(f"   Expected: ({len(X)}, {expected_features})")
    
    # 檢查前幾個序列的時間連續性
    for i in range(min(3, len(X))):  # 檢查前3個序列
        seq = X.iloc[i].values.reshape(window_size, n_features)
        
        # 假設 Close 是第 4 個特徵 (0-indexed: 3)
        close_prices = seq[:, 3]
        
        # 計算價格變化
        price_changes = np.abs(np.diff(close_prices))
        avg_change = np.mean(price_changes)
        max_change = np.max(price_changes)
        
        print(f"\n📊 Sequence {i+1} continuity analysis:")
        print(f"   Close prices (first 5 steps): {close_prices[:5]}")
        print(f"   Average price change: {avg_change:.6f}")
        print(f"   Maximum price change: {max_change:.6f}")
        
        # 標準化後的價格變化應該很小
        if avg_change > 0.05:  # 如果平均變化超過5%，可能有問題
            print(f"   ⚠️  WARNING: Large price jumps detected in sequence {i+1}!")
        else:
            print(f"   ✅ PASS: Temporal continuity verified")
    
    print("="*80 + "\n")
    return True

def inference_single_stock(
    ticker: str,
    model_path: str,
    model_type: str,
    window_size: int = 30,
    forecast_horizon: int = 5,
    start: str = "2013-12-01",
    end: str = "2023-12-31",
    output_dir: str = "output/results",
    scaler_dir: str = "output/scalers"
) -> tuple:
    """
    Run inference on a single stock using the trained global model.
    Uses the SAME scaler from training (not refitted).
    """
    import pickle
    
    print(f"\n{'=' * 80}")
    print(f"Running inference for {ticker}")
    print(f"Model: {model_type}")
    print(f"Model path: {model_path}")
    print(f"{'=' * 80}\n")

    # ✅ Load the scaler from training
    ticker_clean = ticker.replace('.', '')
    scaler_path = os.path.join(scaler_dir, f"{ticker_clean}_scaler.pkl")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}\n"
            f"Please train the model first to generate scaler files."
        )
    
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    
    scaler = scaler_data['scaler']
    existing_numeric = scaler_data['columns']
    train_date_range = scaler_data['date_range']
    
    print(f"✅ Loaded scaler from training period: {train_date_range[0]} to {train_date_range[1]}")
    print(f"   Features: {existing_numeric}")

    # Load and preprocess data
    csv_path = f"data/raw/{ticker.replace('.', '')}.csv"
    data = read_csv(csv_path)

    # Filter by date range
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[(data["Date"] >= start) & (data["Date"] <= end)].copy()

    # Store original Close values for inverse transform
    dates = data["Date"].values

    # ✅ Use the SAME scaler from training (transform only, no fit!)
    if existing_numeric:
        # IMPORTANT: Use transform() NOT fit_transform()
        data[existing_numeric] = scaler.transform(data[existing_numeric])
        print(f"✅ Applied training scaler to inference data (transform only)")
        
        # Show if data is out of training range
        for col in existing_numeric:
            col_min = data[col].min()
            col_max = data[col].max()
            if col_min < 0 or col_max > 1:
                print(f"   ⚠️  {col}: range [{col_min:.3f}, {col_max:.3f}] - outside [0, 1]")
                print(f"      This indicates {ticker} prices are outside training range")

    # Save scaled data temporarily
    temp_csv_path = f"data/raw/temp_{ticker.replace('.', '')}_scaled.csv"
    data.to_csv(temp_csv_path, index=False)

    # Create pipeline
    pipeline = MLPipeline(
        ticker=ticker,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        target_column="Close",
        feature_columns=["Open", "High", "Low", "Close", "Volume"],
        test_size=0.2,
        random_state=42,
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
        "epochs": 100,
    }

    # Add model-specific parameters
    if model_type == "trans_lstm":
        model_params.update({"lstm_hidden_size": 128, "lstm_num_layers": 2})
    elif model_type == "lstm":
        model_params.update({"hidden_size": 128, "num_layers": 2})
    elif model_type == "mlp_lstm":
        model_params.update({
            "mlp_hidden_sizes": [128, 64],
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
        })

    # Run inference pipeline with scaled data
    predictions, actual, inference_dates = pipeline.run_inference_pipeline(
        model_path=model_path,
        model_type=model_type,
        model_params=model_params,
        start=start,
        end=end,
        from_csv=True,
        csv_path=temp_csv_path,
        drop_columns=["Date"],
        handle_missing=True,
        scale_data=False,  # Already scaled above using training scaler
        save_results=False,  # We'll save manually after inverse transform
        output_dir=output_dir,
    )

    # ✅ Inverse transform using the SAME scaler
    close_idx = existing_numeric.index("Close")
    n_features = len(existing_numeric)

    # Predictions
    pred_full = np.zeros((len(predictions), n_features))
    pred_full[:, close_idx] = predictions
    predictions_original = scaler.inverse_transform(pred_full)[:, close_idx]

    # Actual values
    actual_full = np.zeros((len(actual), n_features))
    actual_full[:, close_idx] = actual
    actual_original = scaler.inverse_transform(actual_full)[:, close_idx]

    print(f"\n✅ Inverse transformed predictions back to original {ticker} price scale")

    # Save results
    from datetime import datetime as dt
    
    results_df = pd.DataFrame({
        "Date": inference_dates,
        "Actual": actual_original,
        "Predicted": predictions_original,
        "Error": actual_original - predictions_original,
        "Abs_Error": np.abs(actual_original - predictions_original),
        "Percentage_Error": np.abs((actual_original - predictions_original) / actual_original) * 100,
    })

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_original, predictions_original))
    mae = mean_absolute_error(actual_original, predictions_original)
    mape = np.mean(np.abs((actual_original - predictions_original) / actual_original)) * 100
    r2 = r2_score(actual_original, predictions_original)

    print(f"\nMetrics for {ticker} (original scale):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    ticker_clean = ticker.replace(".", "")
    results_path = os.path.join(
        output_dir, f"{ticker_clean}_{model_type}_predictions_{timestamp}.csv"
    )
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Clean up temp file
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    return predictions_original, actual_original, inference_dates

def inference_multi_stock(
    tickers: list[str],
    model_path: str,
    model_type: str = "transformer",
    window_size: int = 30,
    forecast_horizon: int = 5,
    start: str = "2013-12-01",
    end: str = "2023-12-31",
    output_dir: str = "output/results",
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
                output_dir=output_dir,
            )

            # Calculate metrics for summary
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            mae = mean_absolute_error(actual, predictions)
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            r2 = r2_score(actual, predictions)

            all_metrics.append(
                {"Ticker": ticker, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
            )

        except Exception as e:
            print(f"\nError processing {ticker}: {str(e)}")
            continue

    # Save combined metrics
    if all_metrics:

        combined_metrics = pd.DataFrame(all_metrics)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_metrics_path = os.path.join(
            output_dir, f"all_stocks_{model_type}_metrics_{timestamp}.csv"
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
    model_type = "transformer"  # transformer, trans_lstm, lstm, mlp_lstm
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
            end=end,
        )

    # Run inference (completely independent from training)
    if "inference" in MODE:
        # Generate model path using the same format as save_model
        # Format: {ticker}_{model_class}model_w{window}_h{horizon}.pth
        model_class_name = model_type  # "lstm", "translstm", or "transformer"
        if model_type == "trans_lstm":
            model_class_name = "translstm"
        elif model_type == "mlp_lstm":
            model_class_name = "mlplstm"

        model_filename = (
            f"MultiStock_{model_class_name}model_w{window_size}_h{forecast_horizon}.pth"
        )
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
            output_dir="output/results",
        )


if __name__ == "__main__":
    main()
