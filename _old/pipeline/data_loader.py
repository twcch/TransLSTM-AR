import pandas as pd
import yfinance as yf


class DataLoader:
    def __init__(self, tickers: list, start: str, end: str):
        self.tickers = tickers
        self.start = start
        self.end = end

        self.data = None

    def fetch_stock_data(self) -> None:
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start, end=self.end)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join([str(c) for c in col if c is not None and str(c) != ""])
                    for col in df.columns.values
                ]

            df = df.reset_index()
            df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

            df["Ticker"] = ticker

            self.__to_csv(df, f"data/raw/{ticker.replace('.', '_')}.csv")

    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, skiprows=[1, 2])
        return df

    def __to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        df.to_csv(filepath, index=False)
