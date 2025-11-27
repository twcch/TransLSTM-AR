import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self, use_log_transform=True):
        self.use_log_transform = use_log_transform

    def load_raw_data(self):
        raw_data_path = Path("data/raw/")

        aapl = pd.read_csv(raw_data_path / "AAPL_2013_2023.csv")
        tsmc = pd.read_csv(raw_data_path / "2330.TW_2013_2023.csv")

        return aapl, tsmc

    def merge_stocks(self, df_1, df_2, on="Date", suffixes=("_1", "_2")):
        aapl = pd.read_csv("data/raw/AAPL_2013_2023.csv")
        tsmc = pd.read_csv("data/raw/2330.TW_2013_2023.csv")

        df_merged = pd.merge(
            aapl, tsmc, on="Date", how="outer", suffixes=("_AAPL", "_TSMC")
        )

        df_clean = df_merged.ffill().dropna()

        return df_clean

    def wide_form_to_long_form(self, df):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])  # 確保格式是時間
            df = df.set_index("Date")

        df.columns = df.columns.str.split("_", expand=True)
        df.columns.names = ["Feature", "Ticker"]
        df_long = df.stack(level="Ticker")
        df_long = df_long.reset_index()

        return df_long

    def min_max_scale(
        self, df, feature_cols, date_col="Date", ticker_col="Ticker", split_ratio=0.8
    ):
        data = df.copy()
        data[date_col] = pd.to_datetime(data[date_col])

        # 1. Ticker Label Encoding
        unique_tickers = sorted(data[ticker_col].unique())
        ticker_map = {name: i for i, name in enumerate(unique_tickers)}
        data[f"{ticker_col}_ID"] = data[ticker_col].map(ticker_map)

        # 2. Split Data
        unique_dates = sorted(data[date_col].unique())
        split_idx = int(len(unique_dates) * split_ratio)
        split_date = unique_dates[split_idx]
        print(f"切分日期: {split_date} (前 {split_ratio*100}% 為訓練集)")

        train_df = data[data[date_col] < split_date].copy()
        test_df = data[data[date_col] >= split_date].copy()

        # 3. Group Scaling with Optional Log Transform
        scalers = {}
        scaled_cols = [f"{c}_Scaled" for c in feature_cols]

        # 初始化欄位
        for c in scaled_cols:
            train_df[c] = 0.0
            test_df[c] = 0.0

        for ticker, ticker_id in ticker_map.items():
            train_mask = train_df[f"{ticker_col}_ID"] == ticker_id
            test_mask = test_df[f"{ticker_col}_ID"] == ticker_id

            if train_mask.sum() == 0:
                continue

            # 取出該公司的特徵數據
            train_data_subset = train_df.loc[train_mask, feature_cols].values.astype(
                float
            )
            test_data_subset = test_df.loc[test_mask, feature_cols].values.astype(float)

            # --- [關鍵修改] Log Transform ---
            if self.use_log_transform:
                # 使用 log1p (log(1+x)) 避免 0 的問題
                train_data_subset = np.log1p(train_data_subset)
                # 測試集也要 log，注意順序：先 log 再 scale
                if len(test_data_subset) > 0:
                    test_data_subset = np.log1p(test_data_subset)
            # -------------------------------

            # Fit Scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(train_data_subset)
            scalers[ticker] = scaler

            # Transform
            train_df.loc[train_mask, scaled_cols] = scaler.transform(train_data_subset)
            if test_mask.sum() > 0:
                test_df.loc[test_mask, scaled_cols] = scaler.transform(test_data_subset)

        return train_df, test_df, scalers, ticker_map

    def preprocess(self):
        # Load raw data from CSV files
        aapl, tsmc = self.load_raw_data()

        # Merge the two stock dataframes on the "Date" column
        df_merged = self.merge_stocks(
            aapl, tsmc, on="Date", suffixes=("_AAPL", "_TSMC")
        )

        # Convert the merged dataframe to long form
        df_long = self.wide_form_to_long_form(df_merged)

        # Define feature columns to be scaled
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]
        train_df, test_df, my_scalers, my_map = self.min_max_scale(
            df_long, feature_cols=feature_cols, split_ratio=0.8
        )

        # Save processed training data to CSV
        train_data = train_df[["Date","Ticker","Close","High","Low","Open","Volume"]]
        train_data.to_csv("data/processed/train_data.csv", index=False)

        train_scaled_data = train_df[["Date", "Ticker_ID","Open_Scaled","High_Scaled","Low_Scaled","Close_Scaled","Volume_Scaled"]]
        train_scaled_data.to_csv("data/processed/train_scaled_data.csv", index=False)
        
        test_data = test_df[["Date","Ticker","Close","High","Low","Open","Volume"]]
        test_data.to_csv("data/processed/test_data.csv", index=False)
        
        test_scaled_data = test_df[["Date", "Ticker_ID","Open_Scaled","High_Scaled","Low_Scaled","Close_Scaled","Volume_Scaled"]]
        test_scaled_data.to_csv("data/processed/test_scaled_data.csv", index=False)

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.preprocess()
