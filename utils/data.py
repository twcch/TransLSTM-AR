import os
import pandas as pd
import yfinance as yf


def fetch_yfinace_data(ticker, start, end) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)

    return data

def read_csv(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path, skiprows=[1])
    
    return data

def to_csv(data: pd.DataFrame, output_path: str) -> None:
    """
    Save the given DataFrame to a CSV file.

    :param data: DataFrame to be saved.
    :param output_path: Path where the CSV file will be saved.
    """
    # 取得目錄路徑
    directory = os.path.dirname(output_path)

    # 如果目錄不存在則建立
    if directory:  # 確保不是空字串
        os.makedirs(directory, exist_ok=True)

    data.to_csv(output_path, index=False)
