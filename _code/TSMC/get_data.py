import yfinance as yf
import matplotlib.pyplot as plt

# 設定股票代碼
stock_code = '2330.TW'

# 使用 yfinance 取得歷史股價資訊
stock_data = yf.download(stock_code, start='2013-12-01', end='2023-12-31')

# 顯示資訊
print(stock_data.head())

# 繪製股價走勢圖
stock_data['Close'].plot(title=f'{stock_code}')
plt.show()
# 將資料存成 CSV 檔案
csv_filename = f'{stock_code}_data.csv'
stock_data.to_csv(csv_filename)

print(f'Data saved to {csv_filename}')


# 使用 yfinance 取得歷史股價資訊
stock_data = yf.download(stock_code, start='2017-05-15', end='2022-05-31')

# 顯示資訊
print(stock_data.head())

# 繪製股價走勢圖
stock_data['Close'].plot(title=f'{stock_code}')
plt.show()
# 將資料存成 CSV 檔案
csv_filename = f'{stock_code}_data_5years.csv'
stock_data.to_csv(csv_filename)

print(f'Data saved to {csv_filename}')