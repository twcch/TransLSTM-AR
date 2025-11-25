from core.data.data_loader import DataLoader
from utils.data import read_csv


def load_data(tickers: list[str]):
    for ticker in tickers:
        print("Loading data from source...")
        data_loader = DataLoader(ticker)
        data = data_loader.get_stock_price()
        data_loader.save_to_csv(f"data/raw/{ticker.replace('.', '')}.csv")
        print(f"Data for {ticker} saved to CSV.")


def main():
    # load: download data from source
    MODE = ["load"]

    if "load" in MODE:
        tickers = ["2330.TW", "AAPL"]
        load_data(tickers)
    
    d = read_csv("data/raw/AAPL.csv")
    print(d.head())


if __name__ == "__main__":
    main()
