import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class SequenceBuilder:
    def __init__(self, seq_len=30, pred_len=5):
        self.seq_len = seq_len
        self.pred_len = pred_len

    def prepare_datasets(
        self, 
        df: pd.DataFrame, 
        feature_cols: list,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32,
        target_col='log_close',
        date_col='Date'
    ):
        """準備訓練、驗證和測試的 DataLoader"""
        tickers = df['Ticker'].unique()
        ticker2id = {t: i for i, t in enumerate(tickers)}
        id2ticker = {i: t for i, t in enumerate(tickers)}
        
        train_X_list, train_y_list, train_id_list, train_date_list = [], [], [], []
        val_X_list, val_y_list, val_id_list, val_date_list = [], [], [], []
        test_X_list, test_y_list, test_id_list, test_date_list = [], [], [], []
        
        scalers = {}
        target_scalers = {}  # ✅ 新增：為目標變數單獨保存 scaler 參數
        
        for ticker in tickers:
            ticker_df = df[df['Ticker'] == ticker].sort_values(by=date_col).reset_index(drop=True)
            
            if len(ticker_df) < self.seq_len + self.pred_len:
                print(f"[WARNING] {ticker} data too short ({len(ticker_df)} rows), skipping.")
                continue
            
            total_len = len(ticker_df)
            usable_len = total_len - self.pred_len
            
            train_end = int(usable_len * train_ratio)
            val_end = int(usable_len * (train_ratio + val_ratio))
            
            min_samples = self.seq_len + self.pred_len
            if train_end < min_samples:
                print(f"[WARNING] {ticker} insufficient training data, skipping.")
                continue
            if (val_end - train_end) < min_samples:
                print(f"[WARNING] {ticker} insufficient validation data, skipping.")
                continue
            if (total_len - val_end) < min_samples:
                print(f"[WARNING] {ticker} insufficient test data, skipping.")
                continue
            
            # ========== 處理特徵 ==========
            feature_data = ticker_df[feature_cols].values
            feature_data = np.where(np.isinf(feature_data), np.nan, feature_data)
            
            # 用訓練集的中位數填充 NaN
            for col_idx in range(feature_data.shape[1]):
                col = feature_data[:, col_idx]
                if np.isnan(col).any():
                    train_col = col[:train_end]
                    median_val = np.nanmedian(train_col)
                    if np.isnan(median_val):
                        median_val = 0.0
                    feature_data[:, col_idx] = np.where(np.isnan(col), median_val, col)
            
            # ✅ 只在訓練集上 Fit Scaler
            scaler = StandardScaler()
            scaler.fit(feature_data[:train_end])
            feature_data_scaled = scaler.transform(feature_data)
            scalers[ticker] = scaler
            
            # ========== 處理目標變數（關鍵修正）==========
            target_data = ticker_df[target_col].values
            target_data = np.where(np.isinf(target_data), np.nan, target_data)
            
            if np.isnan(target_data).any():
                train_target = target_data[:train_end]
                median_target = np.nanmedian(train_target)
                if np.isnan(median_target):
                    median_target = 0.0
                target_data = np.where(np.isnan(target_data), median_target, target_data)
            
            # ✅ 為目標變數單獨創建 scaler（使用訓練集的統計量）
            target_scaler = StandardScaler()
            target_scaler.fit(target_data[:train_end].reshape(-1, 1))
            target_data_scaled = target_scaler.transform(target_data.reshape(-1, 1)).flatten()
            
            # ✅ 保存目標變數的 scaler 參數
            target_scalers[ticker] = {
                'mean': float(target_scaler.mean_[0]),
                'scale': float(target_scaler.scale_[0])
            }
            
            print(f"[INFO] {ticker} - Target scaler: mean={target_scalers[ticker]['mean']:.6f}, scale={target_scalers[ticker]['scale']:.6f}")

            # ========== 準備日期 ==========
            dates = ticker_df[date_col].values

            # ========== 訓練集 ==========
            train_data = feature_data_scaled[:train_end + self.pred_len]
            train_target = target_data_scaled[:train_end + self.pred_len]
            train_dates = dates[:train_end + self.pred_len]
            
            train_X, train_y, train_date = self._create_sequences(
                train_data, train_target, train_dates
            )
            
            if len(train_X) > 0:
                train_X_list.append(train_X)
                train_y_list.append(train_y)
                train_id_list.append(np.full(len(train_X), ticker2id[ticker]))
                train_date_list.append(train_date)
                print(f"[INFO] {ticker} Train: {len(train_X)} sequences")
            
            # ========== 驗證集 ==========
            val_data = feature_data_scaled[train_end:val_end + self.pred_len]
            val_target = target_data_scaled[train_end:val_end + self.pred_len]
            val_dates = dates[train_end:val_end + self.pred_len]
            
            val_X, val_y, val_date = self._create_sequences(
                val_data, val_target, val_dates
            )
            
            if len(val_X) > 0:
                val_X_list.append(val_X)
                val_y_list.append(val_y)
                val_id_list.append(np.full(len(val_X), ticker2id[ticker]))
                val_date_list.append(val_date)
                print(f"[INFO] {ticker} Val:   {len(val_X)} sequences")
            
            # ========== 測試集 ==========
            test_data = feature_data_scaled[val_end:]
            test_target = target_data_scaled[val_end:]
            test_dates = dates[val_end:]
            
            test_X, test_y, test_date = self._create_sequences(
                test_data, test_target, test_dates
            )
            
            if len(test_X) > 0:
                test_X_list.append(test_X)
                test_y_list.append(test_y)
                test_id_list.append(np.full(len(test_X), ticker2id[ticker]))
                test_date_list.append(test_date)
                print(f"[INFO] {ticker} Test:  {len(test_X)} sequences")

        # ========== 創建 DataLoader ==========
        train_loader = self._create_loader(
            train_X_list, train_y_list, train_id_list, train_date_list, 
            batch_size, shuffle=False
        )
        val_loader = self._create_loader(
            val_X_list, val_y_list, val_id_list, val_date_list,
            batch_size, shuffle=False
        )
        test_loader = self._create_loader(
            test_X_list, test_y_list, test_id_list, test_date_list,
            batch_size, shuffle=False
        )
        
        # ✅ 將 target_scalers 也一起返回
        scalers['target_scalers'] = target_scalers
        
        print(f"\n[INFO] Total target scalers saved: {len(target_scalers)}")
        
        return train_loader, val_loader, test_loader, scalers, ticker2id, id2ticker

    def _create_sequences(self, data, targets, dates):
        """創建序列數據"""
        X, y, date_list = [], [], []
        
        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            X.append(data[i : i + self.seq_len])
            y.append(targets[i + self.seq_len : i + self.seq_len + self.pred_len])
            
            # 記錄序列的最後一天日期（用於繪圖）
            date_list.append(dates[i + self.seq_len - 1])
            
        return (
            np.array(X, dtype=np.float32), 
            np.array(y, dtype=np.float32),
            np.array(date_list)
        )

    def _create_loader(self, X_list, y_list, id_list, date_list, batch_size, shuffle):
        """創建 DataLoader"""
        if not X_list:
            print("[WARNING] No data for loader")
            return None
        
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        ids = np.concatenate(id_list)
        dates = np.concatenate(date_list)
        
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(ids),
            torch.from_numpy(dates.astype('datetime64[D]').astype(np.int64))  # 轉換日期為整數
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)