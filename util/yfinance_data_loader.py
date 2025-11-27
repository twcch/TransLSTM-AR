import pandas as pd
import yfinance as yf


class YFinanceDataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date

        self.data = None

    def load_data(self):
        print(f"Loading data for {self.ticker} from {self.start} to {self.end}...")
        data = yf.download(self.ticker, start=self.start, end=self.end)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        self.data = data

        print(f"Data loaded: {len(self.data)} records.")

        return self.data

    def save_to_csv(self, file_path):
        if self.data is None:
            raise ValueError(
                "Data not loaded. Please call load_data() before saving to CSV."
            )

        print(f"Saving data to {file_path}...")
        self.data.to_csv(file_path)
        print("Data saved successfully.")


if __name__ == "__main__":
    tickers = ["AAPL", "2330.TW"]
    for ticker in tickers:
        loader = YFinanceDataLoader(ticker, "2013-12-01", "2023-12-31")
        data = loader.load_data()
        loader.save_to_csv(f"data/raw/{ticker}_2013_2023.csv")
