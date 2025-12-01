import numpy as np
import pandas as pd

class FeatureEngineering:
    def __init__(self, data):
        self.data = data.copy()  # ✅ 使用 copy 避免修改原始資料

    def add_features(self):
        """
        為資料集加入技術指標特徵
        
        ⚠️ 關鍵原則：所有特徵都必須使用「昨天及以前」的資訊
        避免資料洩漏 (Data Leakage)
        """
        df = self.data.copy()
        
        # ===== 1. 基礎價格特徵 =====
        df['log_close'] = np.log(df['Close'])
        
        # ✅ 使用 groupby 確保不同股票分開計算
        df['prev_log_close'] = df.groupby('Ticker')['log_close'].shift(1)
        df['prev_high'] = df.groupby('Ticker')['High'].shift(1)
        df['prev_low'] = df.groupby('Ticker')['Low'].shift(1)
        df['prev_open'] = df.groupby('Ticker')['Open'].shift(1)
        df['prev_volume'] = df.groupby('Ticker')['Volume'].shift(1)
        
        # ===== 2. 對數報酬率 =====
        df['log_ret'] = df['log_close'] - df['prev_log_close']
        
        # ===== 3. 成交量變化（安全版本，避免除零）=====
        # ✅ 修正：使用 shift(2) 確保使用「前天的成交量」計算變化率
        df['prev_volume_2'] = df.groupby('Ticker')['Volume'].shift(2)
        df['vol_change'] = df.apply(
            lambda row: (row['prev_volume'] - row['prev_volume_2']) / row['prev_volume_2']
            if pd.notna(row['prev_volume_2']) and row['prev_volume_2'] > 0 
            else 0.0,
            axis=1
        )
        
        # ===== 4. 移動平均線特徵 =====
        for window in [5, 10, 20, 60]:
            # ✅ 關鍵：先 shift(1) 再計算移動平均，確保不使用當天資訊
            shifted_close = df.groupby('Ticker')['Close'].shift(1)
            df[f'ma{window}'] = shifted_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # MA 比率（當前收盤價 / MA）
            df[f'ma{window}_close_ratio'] = df['prev_log_close'] / np.log(df[f'ma{window}'] + 1e-10)
        
        # ===== 5. 價格範圍特徵 =====
        # 高低價差比率
        df['hl_range'] = (df['prev_high'] - df['prev_low']) / (df['prev_high'] + 1e-10)
        
        # 開盤與收盤差比率
        df['oc_range'] = (df['prev_log_close'] - np.log(df['prev_open'] + 1e-10)) / (np.abs(np.log(df['prev_open'] + 1e-10)) + 1e-10)
        
        # ===== 6. RSI (相對強弱指標) =====
        for window in [14]:
            # ✅ 使用 shift(1) 確保使用昨天的價格
            shifted_close = df.groupby('Ticker')['Close'].shift(1)
            delta = shifted_close.groupby(df['Ticker']).diff()
            
            gain = delta.where(delta > 0, 0).groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            loss = -delta.where(delta < 0, 0).groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            rs = gain / (loss + 1e-10)  # 避免除零
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # RSI 標準化到 [-1, 1] 區間
            df[f'rsi_{window}_norm'] = (df[f'rsi_{window}'] - 50) / 50
        
        # ===== 7. MACD (指數平滑異同移動平均線) =====
        shifted_close = df.groupby('Ticker')['Close'].shift(1)
        
        ema12 = shifted_close.groupby(df['Ticker']).transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        ema26 = shifted_close.groupby(df['Ticker']).transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df.groupby('Ticker')['macd'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD 標準化（除以價格）
        df['macd_norm'] = df['macd'] / (shifted_close + 1e-10)
        df['macd_histogram_norm'] = df['macd_histogram'] / (shifted_close + 1e-10)
        
        # ===== 8. 布林通道 (Bollinger Bands) =====
        for window in [20]:
            shifted_close = df.groupby('Ticker')['Close'].shift(1)
            
            rolling_mean = shifted_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            rolling_std = shifted_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            df[f'bb_middle_{window}'] = rolling_mean
            
            # 布林通道位置（0 = 下軌，0.5 = 中軌，1 = 上軌）
            bb_range = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            df[f'bb_position_{window}'] = (
                (df.groupby('Ticker')['Close'].shift(1) - df[f'bb_lower_{window}']) / 
                (bb_range + 1e-10)
            )
            
            # 布林通道寬度（波動率指標）
            df[f'bb_width_{window}'] = bb_range / (rolling_mean + 1e-10)
        
        # ===== 9. ATR (平均真實波幅) =====
        for window in [14]:
            # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            prev_close = df.groupby('Ticker')['Close'].shift(2)  # ✅ 使用前天收盤價
            
            tr1 = df['prev_high'] - df['prev_low']
            tr2 = np.abs(df['prev_high'] - prev_close)
            tr3 = np.abs(df['prev_low'] - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            df[f'atr_{window}'] = true_range.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # ATR 標準化（除以價格）
            df[f'atr_{window}_norm'] = df[f'atr_{window}'] / (df.groupby('Ticker')['Close'].shift(1) + 1e-10)
        
        # ===== 10. 波動率 (Volatility) =====
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df.groupby('Ticker')['log_ret'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # ===== 11. 動量指標 (Momentum) =====
        for window in [5, 10, 20]:
            shifted_close = df.groupby('Ticker')['Close'].shift(1)
            shifted_close_n = df.groupby('Ticker')['Close'].shift(window + 1)
            
            df[f'momentum_{window}'] = (shifted_close - shifted_close_n) / (shifted_close_n + 1e-10)
        
        # ===== 12. 價格位置（相對於近期高低點）=====
        for window in [20, 60]:
            shifted_close = df.groupby('Ticker')['Close'].shift(1)
            
            rolling_high = shifted_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            rolling_low = shifted_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            df[f'price_position_{window}'] = (
                (shifted_close - rolling_low) / 
                (rolling_high - rolling_low + 1e-10)
            )
        
        # ===== 13. 成交量指標 =====
        # 成交量相對於移動平均的比率
        for window in [5, 20]:
            shifted_volume = df.groupby('Ticker')['Volume'].shift(1)
            volume_ma = shifted_volume.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'volume_ratio_{window}'] = shifted_volume / (volume_ma + 1e-10)
        
        # ===== 14. 星期幾特徵（週期性）=====
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek  # 0=週一, 4=週五
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # ===== 15. 清理無限值和異常值 =====
        # 將 inf 和 -inf 替換為 NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 刪除輔助欄位
        df = df.drop(columns=['prev_volume_2'], errors='ignore')
        
        self.data = df
        return self.data

    def drop_na(self):
        """
        刪除 NaN 值
        
        策略：只刪除關鍵欄位的 NaN，保留更多數據
        """
        # 關鍵欄位：這些欄位缺失會導致模型無法訓練
        critical_cols = ['log_close', 'prev_log_close', 'log_ret']
        
        # 檢查哪些欄位存在
        existing_critical_cols = [col for col in critical_cols if col in self.data.columns]
        
        # 只刪除關鍵欄位的 NaN
        original_len = len(self.data)
        self.data = self.data.dropna(subset=existing_critical_cols).reset_index(drop=True)
        dropped_len = original_len - len(self.data)
        
        print(f"[INFO] Dropped {dropped_len} rows with missing critical features ({dropped_len/original_len*100:.2f}%)")
        
        # 對於其他欄位的 NaN，用 0 填充（這些通常是技術指標在計算初期的值）
        self.data = self.data.fillna(0)
        
        return self.data