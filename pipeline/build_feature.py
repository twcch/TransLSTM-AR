import numpy as np
import pandas as pd

class FeatureEngineering:
    def __init__(self, data):
        self.data = data.copy()

    def add_features(self):
        """為資料集加入技術指標特徵"""
        df = self.data.copy()
        
        # ===== 1. 基礎價格特徵（對數尺度）=====
        df['log_close'] = np.log(df['Close'])
        df['log_high'] = np.log(df['High'])
        df['log_low'] = np.log(df['Low'])
        df['log_open'] = np.log(df['Open'])
        
        # Shift 後的對數價格
        df['prev_log_close'] = df.groupby('Ticker')['log_close'].shift(1)
        df['prev_log_high'] = df.groupby('Ticker')['log_high'].shift(1)
        df['prev_log_low'] = df.groupby('Ticker')['log_low'].shift(1)
        df['prev_log_open'] = df.groupby('Ticker')['log_open'].shift(1)
        
        # 成交量（對數尺度）
        df['log_volume'] = np.log(df['Volume'] + 1)
        df['prev_log_volume'] = df.groupby('Ticker')['log_volume'].shift(1)
        
        # ===== 2. 對數報酬率 =====
        df['log_ret'] = df['log_close'] - df['prev_log_close']
        
        # ===== 3. 成交量變化（對數差異）=====
        df['log_vol_change'] = df.groupby('Ticker')['log_volume'].diff()
        
        # ===== 4. 移動平均線特徵（對數尺度）=====
        for window in [5, 10, 20, 60]:
            # ✅ 修正：使用 transform 而不是 apply，避免索引問題
            df[f'log_ma{window}'] = df.groupby('Ticker')['log_close'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # MA 差異
            df[f'log_ma{window}_diff'] = df['prev_log_close'] - df[f'log_ma{window}']
        
        # ===== 5. 價格範圍特徵（對數尺度）=====
        df['log_hl_range'] = df['prev_log_high'] - df['prev_log_low']
        df['log_oc_range'] = df['prev_log_close'] - df['prev_log_open']
        
        # ===== 6. RSI =====
        for window in [14]:
            shifted_log_ret = df.groupby('Ticker')['log_ret'].shift(1)
            
            gain = shifted_log_ret.where(shifted_log_ret > 0, 0).groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            loss = -shifted_log_ret.where(shifted_log_ret < 0, 0).groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            rs = gain / (loss + 1e-10)
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            df[f'rsi_{window}_norm'] = (df[f'rsi_{window}'] - 50) / 50
        
        # ===== 7. MACD =====
        shifted_log_close = df.groupby('Ticker')['log_close'].shift(1)
        
        ema12 = shifted_log_close.groupby(df['Ticker']).transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        ema26 = shifted_log_close.groupby(df['Ticker']).transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )
        
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df.groupby('Ticker')['macd'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ===== 8. 布林通道 =====
        for window in [20]:
            shifted_log_close = df.groupby('Ticker')['log_close'].shift(1)
            
            rolling_mean = shifted_log_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            rolling_std = shifted_log_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            df[f'bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            df[f'bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            
            bb_range = df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']
            df[f'bb_position_{window}'] = (
                (shifted_log_close - df[f'bb_lower_{window}']) / 
                (bb_range + 1e-10)
            )
            df[f'bb_width_{window}'] = bb_range
        
        # ===== 9. ATR =====
        for window in [14]:
            prev_log_close = df.groupby('Ticker')['log_close'].shift(2)
            
            tr1 = df['prev_log_high'] - df['prev_log_low']
            tr2 = np.abs(df['prev_log_high'] - prev_log_close)
            tr3 = np.abs(df['prev_log_low'] - prev_log_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            df[f'atr_{window}'] = true_range.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # ===== 10. 波動率 =====
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df.groupby('Ticker')['log_ret'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # ===== 11. 動量 =====
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = (
                df.groupby('Ticker')['log_close'].shift(1) - 
                df.groupby('Ticker')['log_close'].shift(window + 1)
            )
        
        # ===== 12. 價格位置 =====
        for window in [20, 60]:
            shifted_log_close = df.groupby('Ticker')['log_close'].shift(1)
            
            rolling_high = shifted_log_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            rolling_low = shifted_log_close.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            df[f'price_position_{window}'] = (
                (shifted_log_close - rolling_low) / 
                (rolling_high - rolling_low + 1e-10)
            )
        
        # ===== 13. 成交量比率 =====
        for window in [5, 20]:
            shifted_log_volume = df.groupby('Ticker')['log_volume'].shift(1)
            volume_ma = shifted_log_volume.groupby(df['Ticker']).transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'log_volume_ma_diff_{window}'] = shifted_log_volume - volume_ma
        
        # ===== 14. 時間特徵 =====
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # ===== 15. 清理 =====
        df = df.replace([np.inf, -np.inf], np.nan)
        
        self.data = df
        return self.data

    def drop_na(self):
        critical_cols = ['log_close', 'prev_log_close', 'log_ret']
        existing_critical_cols = [col for col in critical_cols if col in self.data.columns]
        
        original_len = len(self.data)
        self.data = self.data.dropna(subset=existing_critical_cols).reset_index(drop=True)
        dropped_len = original_len - len(self.data)
        
        print(f"[INFO] Dropped {dropped_len} rows ({dropped_len/original_len*100:.2f}%)")
        
        self.data = self.data.fillna(0)
        
        return self.data