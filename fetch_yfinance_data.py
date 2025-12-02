import pandas as pd
import yfinance as yf


def get_data(ticker_symbol, start, end):
    df = yf.download(ticker_symbol, start=start, end=end)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if c is not None and str(c) != ""])
            for col in df.columns.values
        ]

    df = df.reset_index()
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    return df

if __name__ == "__main__":
    tickers = ["2330.TW", "AAPL"]
    start_date = "2013-12-01"
    end_date = "2023-12-31"
    
    for ticker in tickers:
        df = get_data(ticker, start_date, end_date)
        
        df.to_csv(f"data/{ticker.replace('.', '_')}.csv", index=False)

