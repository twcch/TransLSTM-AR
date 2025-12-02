import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# 設定日期範圍
end_date = datetime.today()
start_date = end_date - timedelta(days=60)  # 抓60天內的資料以取得40個交易日

# 加入延遲以避免 Yahoo Finance rate limit
print("等待 5 秒以避免觸發 rate limit...")
time.sleep(5)

# 下載股價資料
df = yf.download("AAPL", start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

# 只保留最近40筆資料
df = df.tail(40)

# 儲存為 Excel 和 CSV 檔案
df.to_csv("AAPL_data_recent.csv")

print("匯出成功：Excel 與 CSV 已儲存。")