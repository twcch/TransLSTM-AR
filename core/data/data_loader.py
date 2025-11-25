import pandas as pd
from utils.data import to_csv, fetch_yfinace_data
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, ticker, start="2013-12-01", end="2023-12-31"):
        self.ticker = ticker
        self.start = start
        self.end = end

        self.data = None

    def get_stock_price(self) -> pd.DataFrame:
        data = fetch_yfinace_data(self.ticker, self.start, self.end)
        data = data.reset_index()

        self.data = data

        return data

    def get_recent_stock_price(
        self, days: int = 60, tail_days: int = 0
    ) -> pd.DataFrame:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        
        end_date = end_date.strftime("%Y-%m-%d")
        start_date = start_date.strftime("%Y-%m-%d")
        data = fetch_yfinace_data(self.ticker, self.start, self.end)
        data = data.reset_index()

        if tail_days > 0:
            data = data.tail(tail_days)

        self.data = data

        return data

    def save_to_csv(self, output_path: str) -> None:
        if self.data is None:
            raise ValueError(
                "No data to save. Please fetch the stock price data first."
            )

        to_csv(self.data, output_path)
