import os
import json
import math
import torch
import pandas as pd
import numpy as np

from pipeline.data_loader import DataLoader
from pipeline.preprocessing import Preprocessor
from pipeline.build_feature import FeatureEngineering
from pipeline.sequence_builder import SequenceBuilder
from pipeline.training import Training

# ✅ 新增: 導入 Encoder-Decoder Transformer
from model.baseline.transformer_model import TransformerEncoderDecoderModel
from model.baseline.lstm_model import LSTMModel
from model.baseline.gru_model import GRUModel


def evaluate_naive_baseline(test_loader, feature_cols, scalers, id2ticker, pred_len=1):
    """
    評估 Naive Baseline: 使用前一天的價格作為預測
    如果模型的 R² 接近這個值，表示只學到了「複製前一天」
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Naive Baseline (Previous Day Prediction)")
    print(f"{'='*60}\n")
    
    all_preds = []
    all_targets = []
    ticker_names = []
    
    for batch_data in test_loader:
        if len(batch_data) == 4:
            X_batch, y_batch, tid_batch, _ = batch_data
        else:
            X_batch, y_batch, tid_batch = batch_data
        
        # 使用序列的最後一個時間步作為預測（log_close 是第 0 個特徵）
        last_day_log_close = X_batch[:, -1, 0].numpy()
        
        # 反標準化 + exp
        for i, tid in enumerate(tid_batch.numpy()):
            ticker = id2ticker[int(tid)]
            scaler = scalers[ticker]
            ticker_names.append(ticker)
            
            # 構造假的特徵向量進行反標準化
            fake_vec = np.zeros((1, len(feature_cols)), dtype=np.float32)
            fake_vec[0, 0] = last_day_log_close[i]
            inv_log = scaler.inverse_transform(fake_vec)[0, 0]
            pred_price = math.exp(inv_log)
            all_preds.append(pred_price)
            
            # 真實值
            fake_true = np.zeros((1, len(feature_cols)), dtype=np.float32)
            if y_batch.dim() == 1:
                fake_true[0, 0] = y_batch[i].item()
            else:
                fake_true[0, 0] = y_batch[i, 0].item()  # 只取 day_1
            inv_true = scaler.inverse_transform(fake_true)[0, 0]
            true_price = math.exp(inv_true)
            all_targets.append(true_price)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 計算指標
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100
    
    # 計算 R²
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # 分股票計算
    print(f"Naive Baseline Results:")
    print(f"  Overall MAPE: {mape:.2f}%")
    print(f"  Overall R²:   {r2:.4f}")
    print(f"\n{'─'*60}")
    
    for ticker in sorted(set(ticker_names)):
        ticker_mask = np.array(ticker_names) == ticker
        ticker_preds = all_preds[ticker_mask]
        ticker_targets = all_targets[ticker_mask]
        
        ticker_mape = np.mean(np.abs((ticker_targets - ticker_preds) / ticker_targets)) * 100
        ticker_ss_res = np.sum((ticker_targets - ticker_preds) ** 2)
        ticker_ss_tot = np.sum((ticker_targets - np.mean(ticker_targets)) ** 2)
        ticker_r2 = 1 - (ticker_ss_res / ticker_ss_tot) if ticker_ss_tot != 0 else 0.0
        
        print(f"[{ticker}]")
        print(f"  MAPE: {ticker_mape:.2f}%")
        print(f"  R²:   {ticker_r2:.4f}")
    
    print(f"\n{'─'*60}")
    print(f"⚠️  If model R² ≈ {r2:.4f}, it only learned to copy yesterday's price!")
    print(f"{'='*60}\n")
    
    return {"mape": mape, "r2": r2}


def main():
    # Options: "load", "preprocessing", "feature_engineering", "train", "test"
    MODE = ["train", "test"]

    # Options: "transformer", "lstm", "gru"
    MODEL_MODE = ["transformer", "lstm", "gru"]

    # Base parameters
    TICKERS = ["2330.TW", "AAPL"]
    START_DATE = "2012-01-01"
    END_DATE = "2024-12-31"

    SEQ_LEN = 60
    LSTM_SEQ_LEN = 45  # LSTM 使用較短序列
    PRED_LEN = 5
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.15
    TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

    BATCH_SIZE = 32

    # ✅ 修改: Transformer 使用 Encoder-Decoder 架構的配置
    MODEL_CONFIGS = {
        "transformer": {
            "learning_rate": 0.00003,      # 維持原學習率
            "epochs": 400,                 # 增加訓練時間
            "early_stopping_patience": 60, # 更大耐心
            "model_params": {
                "d_model": 256,            # 增大到 256
                "nhead": 8,                # 增加到 8 頭
                "num_encoder_layers": 4,   # Encoder N 層
                "num_decoder_layers": 6,   # Decoder N 層
                "dropout": 0.15,            # 降低 dropout
            }
        },
        "lstm": {
            "learning_rate": 0.00005,
            "epochs": 200,
            "early_stopping_patience": 40,
            "model_params": {
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.5,
            }
        },
        "gru": {
            "learning_rate": 0.0002,
            "epochs": 150,
            "early_stopping_patience": 30,
            "model_params": {
                "hidden_dim": 256,
                "num_layers": 3,
                "dropout": 0.25,
            }
        }
    }

    # Device configuration
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"Stock Price Prediction Pipeline")
    print(f"{'='*60}")
    print(f"Mode: {MODE}")
    print(f"Models: {MODEL_MODE}")
    print(f"Device: {device}")
    print(f"Tickers: {TICKERS}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Sequence Length: {SEQ_LEN} (LSTM: {LSTM_SEQ_LEN})")
    print(f"Prediction Length: {PRED_LEN}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    # ========== Data Loading ==========
    if "load" in MODE:
        print(f"{'='*60}")
        print(f"Step 1: Loading Data")
        print(f"{'='*60}\n")

        data_loader = DataLoader(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE)
        raw_data = data_loader.load_data()
        raw_data.to_csv("data/raw/raw_data.csv", index=False)

        print(f"[INFO] Raw data saved to data/raw/raw_data.csv")
        print(f"[INFO] Shape: {raw_data.shape}")
        print(f"[INFO] Columns: {raw_data.columns.tolist()}\n")

    # ========== Preprocessing ==========
    if "preprocessing" in MODE:
        print(f"{'='*60}")
        print(f"Step 2: Preprocessing")
        print(f"{'='*60}\n")

        raw_data = pd.read_csv("data/raw/raw_data.csv")
        preprocessor = Preprocessing()
        processed_data = preprocessor.preprocess(raw_data)
        processed_data.to_csv("data/processed/processed_data.csv", index=False)

        print(f"[INFO] Processed data saved to data/processed/processed_data.csv")
        print(f"[INFO] Shape: {processed_data.shape}\n")

    # ========== Feature Engineering ==========
    if "feature_engineering" in MODE:
        print(f"{'='*60}")
        print(f"Step 3: Feature Engineering")
        print(f"{'='*60}\n")

        processed_data = pd.read_csv("data/processed/processed_data.csv")
        feature_builder = FeatureBuilder()
        featured_data = feature_builder.build_features(processed_data)
        featured_data.to_csv("data/featured/featured_data.csv", index=False)

        print(f"[INFO] Featured data saved to data/featured/featured_data.csv")
        print(f"[INFO] Shape: {featured_data.shape}")
        print(f"[INFO] Features: {featured_data.columns.tolist()}\n")

    # ========== Training & Testing ==========
    if "train" in MODE or "test" in MODE:
        featured_data = pd.read_csv("data/featured/featured_data.csv")
        print(f"[INFO] Loaded featured data: {featured_data.shape}")
        print(f"[INFO] Tickers: {featured_data['Ticker'].unique().tolist()}\n")

        # 定義特徵欄位
        feature_cols = [
            'log_close',
            'High',
            'Low',
            'Open',
            'Volume',
            'log_ret',
            'vol_change',
            'ma5_close_ratio',
            'ma20_close_ratio',
            'hl_range'
        ]

        # 為不同模型準備不同的資料集
        datasets = {}
        
        for model_type in MODEL_MODE:
            # LSTM 使用較短的序列長度
            current_seq_len = LSTM_SEQ_LEN if model_type == "lstm" else SEQ_LEN
            
            sequence_builder = SequenceBuilder(
                seq_len=current_seq_len,
                pred_len=PRED_LEN
            )

            (
                train_loader,
                val_loader,
                test_loader,
                scalers,
                ticker2id,
                id2ticker,
            ) = sequence_builder.prepare_datasets(
                df=featured_data,
                feature_cols=feature_cols,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                batch_size=BATCH_SIZE,
                target_col='log_close',
                date_col='Date'
            )
            
            datasets[model_type] = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'scalers': scalers,
                'ticker2id': ticker2id,
                'id2ticker': id2ticker,
            }
            
            print(f"[INFO] Prepared dataset for {model_type.upper()} with seq_len={current_seq_len}\n")

        input_dim = len(feature_cols)
        all_results = {}

        # ========== Training Phase ==========
        if "train" in MODE:
            for model_type in MODEL_MODE:
                print(f"\n{'='*60}")
                print(f"Training {model_type.upper()} Model")
                print(f"{'='*60}")
                
                config = MODEL_CONFIGS[model_type]
                dataset = datasets[model_type]
                
                print(f"[INFO] Learning Rate: {config['learning_rate']}")
                print(f"[INFO] Max Epochs: {config['epochs']}")
                print(f"[INFO] Early Stopping Patience: {config['early_stopping_patience']}")
                print(f"[INFO] Model Parameters: {config['model_params']}\n")

                # ✅ 修改: 根據模型類型初始化模型
                if model_type == "transformer":
                    # 使用新的 Encoder-Decoder 架構
                    model = TransformerEncoderDecoderModel(
                        input_dim=input_dim,
                        d_model=config["model_params"]["d_model"],
                        nhead=config["model_params"]["nhead"],
                        num_encoder_layers=config["model_params"]["num_encoder_layers"],
                        num_decoder_layers=config["model_params"]["num_decoder_layers"],
                        output_dim=1,
                        pred_len=PRED_LEN,
                        dropout=config["model_params"]["dropout"],
                    ).to(device)

                elif model_type == "lstm":
                    model = LSTMModel(
                        input_dim=input_dim,
                        hidden_dim=config["model_params"]["hidden_dim"],
                        num_layers=config["model_params"]["num_layers"],
                        output_dim=1,
                        pred_len=PRED_LEN,
                        dropout=config["model_params"]["dropout"],
                    ).to(device)

                elif model_type == "gru":
                    model = GRUModel(
                        input_dim=input_dim,
                        hidden_dim=config["model_params"]["hidden_dim"],
                        num_layers=config["model_params"]["num_layers"],
                        output_dim=1,
                        pred_len=PRED_LEN,
                        dropout=config["model_params"]["dropout"],
                    ).to(device)

                else:
                    print(f"[WARNING] Unknown model type: {model_type}")
                    continue

                training = Training(
                    model=None,
                    optimizer=None,
                    loss_fn=None,
                    learning_rate=config["learning_rate"],
                    epochs=config["epochs"],
                    device=device,
                    early_stopping_patience=config["early_stopping_patience"],
                )

                best_epoch = training.train_model(
                    model=model,
                    train_loader=dataset['train_loader'],
                    val_loader=dataset['val_loader'],
                    model_name=model_type.upper(),
                )

                model_save_path = f"output/model/{model_type}_model.pth"
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"\n[INFO] {model_type.upper()} model saved to {model_save_path}"
                )
                print(f"[INFO] Best epoch: {best_epoch}\n")

        # ========== Testing Phase ==========
        if "test" in MODE:
            # 先評估 Naive Baseline (只需跑一次)
            if "transformer" in MODEL_MODE:
                naive_results = evaluate_naive_baseline(
                    test_loader=datasets["transformer"]['test_loader'],
                    feature_cols=feature_cols,
                    scalers=datasets["transformer"]['scalers'],
                    id2ticker=datasets["transformer"]['id2ticker'],
                    pred_len=PRED_LEN
                )
                all_results["naive_baseline"] = naive_results

            for model_type in MODEL_MODE:
                print(f"\n{'='*60}")
                print(f"Testing {model_type.upper()} Model")
                print(f"{'='*60}\n")

                config = MODEL_CONFIGS[model_type]
                dataset = datasets[model_type]

                # ✅ 修改: 初始化模型
                if model_type == "transformer":
                    # 使用新的 Encoder-Decoder 架構
                    model = TransformerEncoderDecoderModel(
                        input_dim=input_dim,
                        d_model=config["model_params"]["d_model"],
                        nhead=config["model_params"]["nhead"],
                        num_encoder_layers=config["model_params"]["num_encoder_layers"],
                        num_decoder_layers=config["model_params"]["num_decoder_layers"],
                        output_dim=1,
                        pred_len=PRED_LEN,
                        dropout=config["model_params"]["dropout"],
                    ).to(device)

                elif model_type == "lstm":
                    model = LSTMModel(
                        input_dim=input_dim,
                        hidden_dim=config["model_params"]["hidden_dim"],
                        num_layers=config["model_params"]["num_layers"],
                        output_dim=1,
                        pred_len=PRED_LEN,
                        dropout=config["model_params"]["dropout"],
                    ).to(device)

                elif model_type == "gru":
                    model = GRUModel(
                        input_dim=input_dim,
                        hidden_dim=config["model_params"]["hidden_dim"],
                        num_layers=config["model_params"]["num_layers"],
                        output_dim=1,
                        pred_len=PRED_LEN,
                        dropout=config["model_params"]["dropout"],
                    ).to(device)

                else:
                    print(f"[WARNING] Unknown model type: {model_type}")
                    continue

                # 載入訓練好的模型
                model_path = f"output/model/{model_type}_model.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"[INFO] Loaded {model_type.upper()} model from {model_path}\n")
                else:
                    print(f"[ERROR] Model file not found: {model_path}")
                    continue

                training = Training(
                    model=None,
                    optimizer=None,
                    loss_fn=None,
                    learning_rate=config["learning_rate"],
                    epochs=config["epochs"],
                    device=device,
                    early_stopping_patience=config["early_stopping_patience"],
                )

                results = training.evaluate_model(
                    model=model,
                    test_loader=dataset['test_loader'],
                    feature_cols=feature_cols,
                    scalers=dataset['scalers'],
                    id2ticker=dataset['id2ticker'],
                    model_name=model_type.upper(),
                    pred_len=PRED_LEN,
                )

                all_results[model_type] = results

            # 儲存所有結果
            results_path = "output/result/test_results.json"
            os.makedirs("output/result", exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=4)

            print(f"\n{'='*60}")
            print(f"All results saved to {results_path}")
            print(f"{'='*60}\n")

            # 顯示模型比較
            print(f"\n{'='*60}")
            print(f"Model Comparison Summary")
            print(f"{'='*60}\n")

            if "naive_baseline" in all_results:
                print(f"Naive Baseline:")
                print(f"  MAPE: {all_results['naive_baseline']['mape']:.2f}%")
                print(f"  R²:   {all_results['naive_baseline']['r2']:.4f}")
                print(f"\n{'─'*60}\n")

            for model_type in MODEL_MODE:
                if model_type in all_results:
                    overall = all_results[model_type]["overall"]
                    print(f"{model_type.upper()}:")
                    print(f"  MAPE:           {overall['mape']:.2f}%")
                    print(f"  R² (weighted):  {overall['r2_weighted']:.4f}")
                    print(f"  R² (mixed):     {overall['r2_mixed']:.4f}")
                    print(f"  RMSE:           {overall['rmse']:.2f}")
                    print(f"  MAE:            {overall['mae']:.2f}")
                    print()

            print(f"{'='*60}\n")


if __name__ == "__main__":
    main()