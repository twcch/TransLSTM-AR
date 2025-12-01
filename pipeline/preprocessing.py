import pandas as pd


class Preprocessor:
    def __init__(self, dataset=list[list]):
        self.dataset = dataset

    def combine_datasets(self):
        """合併多個股票的資料集"""
        combined = pd.concat(self.dataset, ignore_index=True)
        
        # ✅ 確保日期格式正確
        combined['Date'] = pd.to_datetime(combined['Date'])
        
        # ✅ 按股票和日期排序
        combined = combined.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        # ✅ 檢查是否有重複
        duplicates = combined.duplicated(subset=['Ticker', 'Date'], keep='first')
        if duplicates.any():
            print(f"[WARNING] Found {duplicates.sum()} duplicate rows, removing...")
            combined = combined[~duplicates].reset_index(drop=True)
        
        print(f"[INFO] Combined dataset shape: {combined.shape}")
        print(f"[INFO] Date range: {combined['Date'].min()} to {combined['Date'].max()}")
        print(f"[INFO] Tickers: {sorted(combined['Ticker'].unique())}")
        
        return combined
