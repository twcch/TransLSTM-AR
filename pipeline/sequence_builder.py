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
        """
        準備訓練、驗證和測試的 DataLoader，並修正資料洩漏問題。
        
        修正重點：
        1. ✅ 先按時間分割原始資料（不重疊）
        2. ✅ 只用訓練集 fit scaler
        3. ✅ 分別對三個集合建立序列（確保時間不重疊）
        4. ✅ 清理無限值和 NaN
        5. ✅ 訓練集不 shuffle（保持時間順序）
        """
        tickers = df['Ticker'].unique()
        ticker2id = {t: i for i, t in enumerate(tickers)}
        id2ticker = {i: t for i, t in enumerate(tickers)}
        
        train_X_list, train_y_list, train_id_list = [], [], []
        val_X_list, val_y_list, val_id_list = [], [], []
        test_X_list, test_y_list, test_id_list = [], [], []
        
        scalers = {}
        
        # 嘗試找到與 target 對應的 feature 欄位索引 (用於標準化 target)
        target_feature_idx = -1
        if 'prev_' + target_col in feature_cols:
             target_feature_idx = feature_cols.index('prev_' + target_col)
        elif target_col in feature_cols:
             target_feature_idx = feature_cols.index(target_col)
        
        for ticker in tickers:
            # 1. 取得該股票資料並排序
            ticker_df = df[df['Ticker'] == ticker].sort_values(by=date_col).reset_index(drop=True)
            
            if len(ticker_df) < self.seq_len + self.pred_len:
                print(f"[WARNING] {ticker} data too short, skipping.")
                continue
            
            # 2. ✅ 修正：計算分割點（確保不重疊）
            total_len = len(ticker_df)
            
            # 為了避免重疊，需要預留 buffer
            # buffer = seq_len，確保驗證集和測試集的第一個序列完全在該時段內
            buffer_size = self.seq_len
            
            # 計算實際可用的資料長度（扣除最後一個序列需要的 pred_len）
            usable_len = total_len - self.pred_len
            
            # 分割點（基於可用長度）
            train_end = int(usable_len * train_ratio)
            val_end = int(usable_len * (train_ratio + val_ratio))
            
            # 確保每個集合都有足夠的資料建立至少一個序列
            min_samples = self.seq_len + self.pred_len
            if train_end < min_samples:
                print(f"[WARNING] {ticker} training set too short, skipping.")
                continue
            if (val_end - train_end) < min_samples:
                print(f"[WARNING] {ticker} validation set too short, skipping.")
                continue
            if (total_len - val_end) < min_samples:
                print(f"[WARNING] {ticker} test set too short, skipping.")
                continue
            
            # 3. ✅ 提取特徵矩陣並清理無限值
            feature_data = ticker_df[feature_cols].values
            
            # 將 inf 替換為 NaN
            feature_data = np.where(np.isinf(feature_data), np.nan, feature_data)
            
            # 用訓練集的中位數填充 NaN（避免使用未來資訊）
            for col_idx in range(feature_data.shape[1]):
                col = feature_data[:, col_idx]
                if np.isnan(col).any():
                    # ✅ 只使用訓練集的統計量
                    train_col = col[:train_end]
                    median_val = np.nanmedian(train_col)
                    if np.isnan(median_val):
                        median_val = 0.0
                    feature_data[:, col_idx] = np.where(np.isnan(col), median_val, col)
            
            # 4. ✅ 只在訓練集上 Fit Scaler
            scaler = StandardScaler()
            scaler.fit(feature_data[:train_end])
            
            # 用訓練集的參數 transform 整份資料
            feature_data_scaled = scaler.transform(feature_data)
            scalers[ticker] = scaler
            
            # 5. 處理 Target 的標準化
            target_data = ticker_df[target_col].values
            
            # 清理 target 的無限值
            target_data = np.where(np.isinf(target_data), np.nan, target_data)
            if np.isnan(target_data).any():
                train_target = target_data[:train_end]
                median_target = np.nanmedian(train_target)
                if np.isnan(median_target):
                    median_target = 0.0
                target_data = np.where(np.isnan(target_data), median_target, target_data)
            
            if target_feature_idx >= 0:
                mean = scaler.mean_[target_feature_idx]
                scale = scaler.scale_[target_feature_idx]
                target_data_scaled = (target_data - mean) / scale
            else:
                # 如果找不到對應特徵，單獨訓練一個 scaler
                target_scaler = StandardScaler()
                target_scaler.fit(target_data[:train_end].reshape(-1, 1))
                target_data_scaled = target_scaler.transform(target_data.reshape(-1, 1)).flatten()
                print(f"[INFO] {ticker}: Using separate scaler for target.")

            # 6. ✅ 修正：分別對三個**不重疊**的時間段建立序列
            
            # 訓練集：從 0 到 train_end + pred_len（確保最後一個序列完整）
            train_data = feature_data_scaled[:train_end + self.pred_len]
            train_target = target_data_scaled[:train_end + self.pred_len]
            train_X, train_y = self._create_sequences(train_data, train_target)
            
            if len(train_X) > 0:
                train_X_list.append(train_X)
                train_y_list.append(train_y)
                train_id_list.append(np.full(len(train_X), ticker2id[ticker]))
                print(f"[INFO] {ticker} Train: {len(train_X)} sequences (data range: 0 to {train_end + self.pred_len})")
            
            # 驗證集：從 train_end 到 val_end + pred_len
            # ✅ 關鍵修正：不往前取，直接從 train_end 開始
            val_data = feature_data_scaled[train_end:val_end + self.pred_len]
            val_target = target_data_scaled[train_end:val_end + self.pred_len]
            val_X, val_y = self._create_sequences(val_data, val_target)
            
            if len(val_X) > 0:
                val_X_list.append(val_X)
                val_y_list.append(val_y)
                val_id_list.append(np.full(len(val_X), ticker2id[ticker]))
                print(f"[INFO] {ticker} Val:   {len(val_X)} sequences (data range: {train_end} to {val_end + self.pred_len})")
            
            # 測試集：從 val_end 到結尾
            test_data = feature_data_scaled[val_end:]
            test_target = target_data_scaled[val_end:]
            test_X, test_y = self._create_sequences(test_data, test_target)
            
            if len(test_X) > 0:
                test_X_list.append(test_X)
                test_y_list.append(test_y)
                test_id_list.append(np.full(len(test_X), ticker2id[ticker]))
                print(f"[INFO] {ticker} Test:  {len(test_X)} sequences (data range: {val_end} to {total_len})")

        # 7. ✅ 建立 DataLoader（訓練集也不 shuffle）
        train_loader = self._create_loader(train_X_list, train_y_list, train_id_list, batch_size, shuffle=False)
        val_loader = self._create_loader(val_X_list, val_y_list, val_id_list, batch_size, shuffle=False)
        test_loader = self._create_loader(test_X_list, test_y_list, test_id_list, batch_size, shuffle=False)
        
        # 檢查是否成功建立資料集
        if train_loader is None:
            print("[ERROR] No training data available!")
        else:
            print(f"\n[INFO] Total training batches: {len(train_loader)}")
        if val_loader is None:
            print("[WARNING] No validation data available!")
        else:
            print(f"[INFO] Total validation batches: {len(val_loader)}")
        if test_loader is None:
            print("[WARNING] No test data available!")
        else:
            print(f"[INFO] Total test batches: {len(test_loader)}\n")
        
        return train_loader, val_loader, test_loader, scalers, ticker2id, id2ticker

    def _create_sequences(self, data, targets):
        """
        建立時間序列資料
        X: [t, t+1, ..., t+seq_len-1]
        y: [t+seq_len, ..., t+seq_len+pred_len-1]
        """
        X, y = [], []
        # 確保有足夠的資料長度來建立一個完整的序列和預測區間
        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            X.append(data[i : i + self.seq_len])
            y.append(targets[i + self.seq_len : i + self.seq_len + self.pred_len])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _create_loader(self, X_list, y_list, id_list, batch_size, shuffle):
        if not X_list:
            return None
        
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        ids = np.concatenate(id_list)
        
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
            torch.from_numpy(ids)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)