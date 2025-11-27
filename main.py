import pandas as pd

aapl = pd.read_csv("data/raw/AAPL_2013_2023.csv")
tsmc = pd.read_csv("data/raw/2330.TW_2013_2023.csv")

df_merged = pd.merge(
    aapl,
    tsmc,
    on="Date",
    how="outer",
    suffixes=("_AAPL", "_TSMC")
)

df_clean = df_merged.ffill().dropna()

print(df_clean.head())
