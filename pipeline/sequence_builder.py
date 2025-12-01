import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ticker_ids, dates=None):
        """
        Args:
            X: 特徵序列 (num_samples, seq_len, num_features)
            y: 目標值 (num_samples,)
            ticker_ids: 股票代號 ID (num_samples,)
            dates: 日期資訊 (num_samples,) - 可選
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)
        self.dates = dates

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.dates is not None:
            return self.X[idx], self.y[idx], self.ticker_ids[idx], self.dates[idx]
        return self.X[idx], self.y[idx], self.ticker_ids[idx]


class SequenceBuilder:
    def __init__(self, seq_len=30, pred_len=1):
        """
        Args:
            seq_len: 序列長度（輸入時間步數）
            pred_len: 預測長度（目前固定為1）
        """
        self.seq_len = seq_len
        self.pred_len = pred_len

    def create_sequences(self, df, features, target='log_close', date_col='Date'):
        """
        從 DataFrame 建立時間序列（支援多步預測）
        
        Args:
            df: 單一股票的 DataFrame
            features: 特徵欄位列表
            target: 目標欄位
            date_col: 日期欄位名稱
        
        Returns:
            X, y, dates
        """
        if len(df) < self.seq_len + self.pred_len:
            return np.array([]), np.array([]), None
        
        # 提取特徵和目標
        feature_values = df[features].values
        target_values = df[target].values
        
        # 提取日期（如果存在）
        dates = None
        if date_col in df.columns:
            dates = df[date_col].values
        
        X_list = []
        y_list = []
        date_list = []
        
        for i in range(len(df) - self.seq_len - self.pred_len + 1):
            # 輸入序列: [i, i+seq_len)
            X_list.append(feature_values[i:i + self.seq_len])
            
            # 目標: 未來 pred_len 天的 log_close
            # shape: (pred_len,)
            y_list.append(target_values[i + self.seq_len : i + self.seq_len + self.pred_len])
            
            # 對應的日期（預測目標的最後一天）
            if dates is not None:
                date_list.append(dates[i + self.seq_len + self.pred_len - 1])
        
        X = np.array(X_list)
        y = np.array(y_list)  # shape: (num_samples, pred_len)
        dates_out = np.array(date_list) if date_list else None
        
        return X, y, dates_out

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
        準備訓練、驗證、測試資料集（支援日期）
        
        Args:
            df: 包含所有股票的 DataFrame
            feature_cols: 特徵欄位列表
            train_ratio: 訓練集比例
            val_ratio: 驗證集比例
            batch_size: batch 大小
            target_col: 目標欄位
            date_col: 日期欄位
        
        Returns:
            train_loader, val_loader, test_loader, scalers, ticker2id, id2ticker
        """
        # 建立股票代號映射
        tickers = df['Ticker'].unique()
        ticker2id = {ticker: idx for idx, ticker in enumerate(tickers)}
        id2ticker = {idx: ticker for ticker, idx in ticker2id.items()}
        
        # 清理無效數據
        print(f"\n[INFO] Cleaning data...")
        print(f"Original data shape: {df.shape}")
        
        # 將 inf/-inf 替換為 NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 檢查每個特徵的 NaN 數量
        nan_counts = df[feature_cols].isna().sum()
        if nan_counts.any():
            print(f"[WARNING] Found NaN values in features:")
            for col in feature_cols:
                if nan_counts[col] > 0:
                    print(f"  {col}: {nan_counts[col]} NaNs")
        
        # 刪除包含 NaN 的行
        df = df.dropna(subset=feature_cols + [target_col])
        print(f"After cleaning: {df.shape}")
        
        # 為每檔股票分別標準化
        scalers = {}
        normalized_dfs = []
        
        for ticker in tickers:
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # 確保日期欄位存在且為字串格式
            if date_col in ticker_df.columns:
                ticker_df[date_col] = ticker_df[date_col].astype(str)
            
            # 檢查該股票是否有足夠的數據
            if len(ticker_df) < self.seq_len + self.pred_len:
                print(f"[WARNING] {ticker} has insufficient data ({len(ticker_df)} rows), skipping...")
                continue
            
            # 標準化
            scaler = StandardScaler()
            ticker_df[feature_cols] = scaler.fit_transform(ticker_df[feature_cols])
            
            scalers[ticker] = scaler
            normalized_dfs.append(ticker_df)
        
        if not normalized_dfs:
            raise ValueError("No valid data remaining after cleaning!")
        
        df_normalized = pd.concat(normalized_dfs, ignore_index=True)
        
        # 分割訓練/驗證/測試集
        X_train_list, y_train_list, tid_train_list, dates_train_list = [], [], [], []
        X_val_list, y_val_list, tid_val_list, dates_val_list = [], [], [], []
        X_test_list, y_test_list, tid_test_list, dates_test_list = [], [], [], []
        
        for ticker in tickers:
            if ticker not in scalers:
                continue
                
            ticker_df = df_normalized[df_normalized['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values(date_col).reset_index(drop=True)
            ticker_id = ticker2id[ticker]
            
            n = len(ticker_df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            # 訓練集
            train_df = ticker_df.iloc[:train_end]
            X_train, y_train, dates_train = self.create_sequences(
                train_df, feature_cols, target_col, date_col
            )
            if len(X_train) > 0:
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                tid_train_list.extend([ticker_id] * len(X_train))
                if dates_train is not None:
                    dates_train_list.extend(dates_train)
            
            # 驗證集
            val_df = ticker_df.iloc[train_end:val_end]
            X_val, y_val, dates_val = self.create_sequences(
                val_df, feature_cols, target_col, date_col
            )
            if len(X_val) > 0:
                X_val_list.append(X_val)
                y_val_list.append(y_val)
                tid_val_list.extend([ticker_id] * len(X_val))
                if dates_val is not None:
                    dates_val_list.extend(dates_val)
            
            # 測試集
            test_df = ticker_df.iloc[val_end:]
            X_test, y_test, dates_test = self.create_sequences(
                test_df, feature_cols, target_col, date_col
            )
            if len(X_test) > 0:
                X_test_list.append(X_test)
                y_test_list.append(y_test)
                tid_test_list.extend([ticker_id] * len(X_test))
                if dates_test is not None:
                    dates_test_list.extend(dates_test)
        
        # 合併所有股票的資料
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        tid_train = np.array(tid_train_list)
        dates_train = np.array(dates_train_list) if dates_train_list else None
        
        X_val = np.concatenate(X_val_list)
        y_val = np.concatenate(y_val_list)
        tid_val = np.array(tid_val_list)
        dates_val = np.array(dates_val_list) if dates_val_list else None
        
        X_test = np.concatenate(X_test_list)
        y_test = np.concatenate(y_test_list)
        tid_test = np.array(tid_test_list)
        dates_test = np.array(dates_test_list) if dates_test_list else None
        
        print(f"\n{'='*60}")
        print(f"Dataset Preparation Summary")
        print(f"{'='*60}")
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples:   {len(X_val)}")
        print(f"Test samples:  {len(X_test)}")
        print(f"Sequence length: {self.seq_len}")
        print(f"Prediction horizon: {self.pred_len}")
        print(f"Number of features: {len(feature_cols)}")
        print(f"Number of stocks: {len(scalers)}")
        print(f"Date information: {'✓ Enabled' if dates_train is not None else '✗ Disabled'}")
        print(f"{'='*60}\n")
        
        # 建立 DataLoader
        train_dataset = TimeSeriesDataset(X_train, y_train, tid_train, dates_train)
        val_dataset = TimeSeriesDataset(X_val, y_val, tid_val, dates_val)
        test_dataset = TimeSeriesDataset(X_test, y_test, tid_test, dates_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        return (
            train_loader,
            val_loader,
            test_loader,
            scalers,
            ticker2id,
            id2ticker,
        )