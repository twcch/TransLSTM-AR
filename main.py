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

from model.baseline.transformer_model import TransformerEncoderDecoderModel
from model.baseline.lstm_model import LSTMModel
from model.baseline.gru_model import GRUModel


def evaluate_naive_baseline(test_loader, feature_cols, scalers, id2ticker, pred_len=5):
    """評估 Naive Baseline"""
    print(f"\n{'='*60}")
    print(f"Evaluating Naive Baseline (Previous Day Prediction)")
    print(f"{'='*60}\n")
    
    target_scalers = scalers.get('target_scalers', {})
    
    if not target_scalers:
        print("[ERROR] No target_scalers found!")
        return {"mape": float('inf'), "r2": -float('inf')}
    
    all_preds = []
    all_targets = []
    ticker_names = []
    
    for batch_data in test_loader:
        X_batch, y_batch, tid_batch = batch_data[:3]
        
        for i, tid in enumerate(tid_batch.numpy()):
            ticker = id2ticker[int(tid)]
            target_mean = target_scalers[ticker]['mean']
            target_scale = target_scalers[ticker]['scale']
            
            # ✅ 方法 1：從 y_batch 的前一個時間步推算
            # 因為 y_batch[0] 是序列結束後的第一天，我們需要「序列最後一天」的價格
            # 這個資訊在 sequence_builder 中並沒有保存，所以我們用另一個方法
            
            # ✅ 方法 2：使用 log_ret 反推
            # log_ret = log_close[t] - log_close[t-1]
            # 所以 log_close[t-1] = log_close[t] - log_ret[t]
            
            # 獲取序列最後一天的 log_ret（標準化後）
            if 'log_ret' not in feature_cols:
                print("[ERROR] 'log_ret' not in feature_cols!")
                return {"mape": float('inf'), "r2": -float('inf')}
            
            log_ret_idx = feature_cols.index('log_ret')
            last_log_ret_scaled = X_batch[i, -1, log_ret_idx].item()
            
            # ✅ 方法 3：直接從 y_batch 的第一天預測往前推一天
            # 但這樣會有 data leakage，因為 y_batch[0] 是未來的資訊
            
            # ✅ 最佳方法：直接使用 y_batch[0] 當作「當前價格」，但這是測試集的第一天
            # Naive Baseline 應該是：pred[t+1] = actual[t]
            # 也就是：pred[day=0] = 序列最後一天的價格（這個資訊我們沒有）
            
            # ✅ 實際可行方法：使用「前一個 batch 的最後一天」
            # 但這需要重構 sequence_builder
            
            # ✅ 暫時方案：用第一天的預測當作 baseline（假設價格不變）
            # pred[day=1] = actual[day=0]
            # pred[day=2] = actual[day=1]
            # ...
            
            for day in range(pred_len):
                # Naive: 預測 = 上一天的真實價格
                if day == 0:
                    # 第一天：我們沒有「序列後的前一天」資訊
                    # 使用序列內最後一天的資訊（但這不完全準確）
                    # 最簡單的方式：直接用 y_batch[0] 當預測（意味著完美預測）
                    # 但這會讓 Naive Baseline 看起來很好
                    
                    # ✅ 更合理的方式：用 y_batch[0] 當作「當前已知價格」
                    # 然後預測接下來的日子都是這個價格
                    y_scaled_day0 = y_batch[i, 0].item()
                    y_log_day0 = y_scaled_day0 * target_scale + target_mean
                    reference_price = np.exp(y_log_day0)
                    
                    # 所有未來天數都預測為 day 0 的價格
                    for future_day in range(pred_len):
                        all_preds.append(reference_price)
                        
                        y_scaled = y_batch[i, future_day].item()
                        y_log = y_scaled * target_scale + target_mean
                        true_price = np.exp(y_log)
                        all_targets.append(true_price)
                        ticker_names.append(ticker)
                    
                    break  # 已經處理完所有天數
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 過濾無效值
    valid_mask = (all_targets > 0) & (all_preds > 0) & np.isfinite(all_preds) & np.isfinite(all_targets)
    all_preds = all_preds[valid_mask]
    all_targets = all_targets[valid_mask]
    ticker_names = np.array(ticker_names)[valid_mask]
    
    if len(all_preds) == 0:
        print("[ERROR] No valid predictions!")
        return {"mape": float('inf'), "r2": -float('inf')}
    
    # 計算指標
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    print(f"Naive Baseline Results:")
    print(f"  Overall MAPE: {mape:.2f}%")
    print(f"  Overall R²:   {r2:.4f}")
    print(f"\n{'─'*60}")
    
    # 分股票結果
    for ticker in sorted(set(ticker_names)):
        ticker_mask = ticker_names == ticker
        ticker_preds = all_preds[ticker_mask]
        ticker_targets = all_targets[ticker_mask]
        
        if len(ticker_preds) == 0:
            continue
            
        ticker_mape = np.mean(np.abs((ticker_targets - ticker_preds) / ticker_targets)) * 100
        ticker_ss_res = np.sum((ticker_targets - ticker_preds) ** 2)
        ticker_ss_tot = np.sum((ticker_targets - np.mean(ticker_targets)) ** 2)
        ticker_r2 = 1 - (ticker_ss_res / ticker_ss_tot) if ticker_ss_tot != 0 else 0.0
        
        print(f"[{ticker}] MAPE: {ticker_mape:.2f}%, R²: {ticker_r2:.4f}")
    
    print(f"\n{'─'*60}")
    print(f"⚠️  Naive Baseline: Predict future = first day of target window")
    print(f"⚠️  If model R² ≈ {r2:.4f}, model didn't learn anything useful!")
    print(f"{'='*60}\n")
    
    return {"mape": float(mape), "r2": float(r2)}


def main():
    PREDICTION_MODE = "autoregressive"
    
    # "load", "preprocessing", "feature_engineering", "train", "test"
    MODE = ["load", "preprocessing", "feature_engineering", "train", "test"]
    MODEL_MODE = ["transformer", "lstm", "gru"]

    TICKERS = ["2330.TW", "AAPL"]
    START_DATE = "2012-01-01"
    END_DATE = "2024-12-31"

    SEQ_LEN = 60
    PRED_LEN = 5
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.15

    BATCH_SIZE = 64

    # ✅ 修正：參數與模型實作一致（降低複雜度以避免過擬合與梯度問題）
    MODEL_CONFIGS = {
        "transformer": {
            "learning_rate": 0.001,
            "epochs": 200,
            "early_stopping_patience": 40,
            "model_params": {
                "d_model": 128,        # 降維：256 -> 128
                "nhead": 4,
                "num_encoder_layers": 3, # 減少層數：6 -> 3
                "num_decoder_layers": 3, # 減少層數：6 -> 3
                "dropout": 0.15,
            }
        },
        "lstm": {
            "learning_rate": 0.001,  # ✅ 提高學習率
            "epochs": 300,
            "early_stopping_patience": 50,
            "model_params": {
                "hidden_dim": 128,   # ✅ 增加容量
                "num_layers": 2,     # ✅ 增加層數
                "dropout": 0.2,      # ✅ 加入 Dropout
                "bidirectional": False,
            }
        },
        "gru": {
            "learning_rate": 0.001,
            "epochs": 200,
            "early_stopping_patience": 40,
            "model_params": {
                "hidden_dim": 64,      # 降維：256 -> 64
                "num_layers": 1,       # 修正：4 -> 1
                "dropout": 0.0,        # 修正：0.2 -> 0.0
            }
        }
    }

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
    print(f"Prediction Mode: {PREDICTION_MODE.upper()}")
    print(f"Device: {device}")
    print(f"Tickers: {TICKERS}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Seq Len: {SEQ_LEN}")
    print(f"Pred Len: {PRED_LEN}")
    print(f"{'='*60}\n")

    if "load" in MODE:
        print(f"{'='*60}")
        print(f"Step 1: Loading Data")
        print(f"{'='*60}\n")

        data_loader = DataLoader(tickers=TICKERS, start=START_DATE, end=END_DATE)
        raw_data = data_loader.fetch_stock_data()
        
        print(f"[INFO] Raw data saved\n")

    if "preprocessing" in MODE:
        print(f"{'='*60}")
        print(f"Step 2: Preprocessing")
        print(f"{'='*60}\n")

        dataset = []
        raw_2330 = pd.read_csv("data/raw/2330_TW.csv")
        dataset.append(raw_2330)
        raw_aapl = pd.read_csv("data/raw/AAPL.csv")
        dataset.append(raw_aapl)
        
        preprocessor = Preprocessor(dataset)
        processed_data = preprocessor.combine_datasets()
        
        os.makedirs("data/processed", exist_ok=True)
        processed_data.to_csv("data/processed/combined_data.csv", index=False)
        print(f"[INFO] Processed data saved\n")

    if "feature_engineering" in MODE:
        print(f"{'='*60}")
        print(f"Step 3: Feature Engineering")
        print(f"{'='*60}\n")

        processed_data = pd.read_csv("data/processed/combined_data.csv")
        feature_builder = FeatureEngineering(processed_data)
        featured_data = feature_builder.add_features()
        featured_data = feature_builder.drop_na()
        
        os.makedirs("data/featured", exist_ok=True)
        featured_data.to_csv("data/featured/featured_data.csv", index=False)
        print(f"[INFO] Featured data saved\n")

    if "train" in MODE or "test" in MODE:
        featured_data = pd.read_csv("data/featured/featured_data.csv")
        print(f"[INFO] Loaded featured data: {featured_data.shape}\n")

        feature_cols = [
            # 基礎特徵
            #'prev_log_close', 
            'prev_log_high', 'prev_log_low', 'prev_log_open', 'prev_log_volume',
            
            # 報酬率與成交量
            'log_ret', 'log_vol_change',
            
            # 價格範圍
            'log_hl_range', 'log_oc_range',
            
            # 移動平均差異
            'log_ma5_diff', 'log_ma10_diff', 'log_ma20_diff', 'log_ma60_diff',
            
            # RSI
            'rsi_14_norm',
            
            # MACD
            'macd', 'macd_histogram',
            
            # 布林通道
            'bb_position_20', 'bb_width_20',
            
            # ATR
            'atr_14',
            
            # 波動率
            'volatility_5', 'volatility_10', 'volatility_20',
            
            # 動量
            'momentum_5', 'momentum_10', 'momentum_20',
            
            # 價格位置
            'price_position_20', 'price_position_60',
            
            # 成交量差異
            'log_volume_ma_diff_5', 'log_volume_ma_diff_20',
            
            # 時間特徵
            'day_of_week', 'is_monday', 'is_friday',
        ]

        datasets = {}
        
        for model_type in MODEL_MODE:
            current_seq_len = SEQ_LEN
            
            sequence_builder = SequenceBuilder(seq_len=current_seq_len, pred_len=PRED_LEN)

            (
                train_loader, val_loader, test_loader,
                scalers, ticker2id, id2ticker,
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
            
            print(f"[INFO] Prepared {model_type.upper()} dataset (seq_len={current_seq_len})\n")

        input_dim = len(feature_cols)
        all_results = {}

        if "train" in MODE:
            for model_type in MODEL_MODE:
                print(f"\n{'='*60}")
                print(f"Training {model_type.upper()}")
                print(f"{'='*60}\n")
                
                config = MODEL_CONFIGS[model_type]
                dataset = datasets[model_type]

                if model_type == "transformer":
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
                        bidirectional=config["model_params"]["bidirectional"],
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

                os.makedirs("output/model", exist_ok=True)
                torch.save(model.state_dict(), f"output/model/{model_type}_model.pth")
                print(f"[INFO] Model saved (best epoch: {best_epoch})\n")

        if "test" in MODE:
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
                print(f"Testing {model_type.upper()} ({PREDICTION_MODE.upper()} Mode)")
                print(f"{'='*60}\n")

                config = MODEL_CONFIGS[model_type]
                dataset = datasets[model_type]

                if model_type == "transformer":
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
                        bidirectional=config["model_params"]["bidirectional"],
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
                    continue

                model_path = f"output/model/{model_type}_model.pth"
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"[INFO] Loaded model from {model_path}\n")
                else:
                    print(f"[ERROR] Model not found: {model_path}\n")
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

                if PREDICTION_MODE == "autoregressive":
                    results = training.evaluate_model_autoregressive(
                        model=model,
                        test_loader=dataset['test_loader'],
                        feature_cols=feature_cols,
                        scalers=dataset['scalers'],
                        id2ticker=dataset['id2ticker'],
                        model_name=model_type.upper(),
                        pred_len=PRED_LEN,
                    )
                else:
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

            results_path = f"output/result/test_results_{PREDICTION_MODE}.json"
            os.makedirs("output/result", exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=4)

            print(f"\n{'='*60}")
            print(f"Results saved to {results_path}")
            print(f"{'='*60}\n")

            print(f"\n{'='*60}")
            print(f"Model Comparison ({PREDICTION_MODE.upper()} Mode)")
            print(f"{'='*60}\n")

            if "naive_baseline" in all_results:
                print(f"Naive Baseline:")
                print(f"  MAPE: {all_results['naive_baseline']['mape']:.2f}%")
                print(f"  R²:   {all_results['naive_baseline']['r2']:.4f}\n")

            for model_type in MODEL_MODE:
                if model_type in all_results:
                    overall = all_results[model_type]["overall"]
                    print(f"{model_type.upper()}:")
                    print(f"  MAPE:      {overall['mape']:.2f}%")
                    print(f"  R² (w):    {overall['r2_weighted']:.4f}")
                    print(f"  RMSE:      {overall['rmse']:.2f}")
                    print(f"  MAE:       {overall['mae']:.2f}\n")

            print(f"{'='*60}\n")


if __name__ == "__main__":
    main()