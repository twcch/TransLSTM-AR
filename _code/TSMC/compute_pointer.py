import pandas as pd
# 計算20日指數移動平均線
def calculate_EMA20(df):
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_MACD(df):
    # 計算12日和26日EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # 計算MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # 計算信號線（9日EMA）
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def calculate_upandlow(df):
    # 計算20日移動平均線和標準差
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()

    # 計算布林帶
    df['Upper_Band'] = df['SMA_20'] + (2 * df['STD_20'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['STD_20'])
    return df

def calculate_bias(df, window):
    df['SMA'] = df['Close'].rolling(window=window).mean()  # 計算移動平均線
    df['Bias'+str(window)] = (df['Close'] - df['SMA']) / df['SMA'] * 100  # 計算乖離率
    return df

# 定義計算KD值的函數
def calculate_kd(data, n=9):
    data['Low_n'] = data['Low'].rolling(window=n).min()  # 計算n天內的最低價
    data['High_n'] = data['High'].rolling(window=n).max()  # 計算n天內的最高價
    data['K'] = 100 * ((data['Close'] - data['Low_n']) / (data['High_n'] - data['Low_n']))  # 計算K值
    data['D'] = data['K'].rolling(window=3).mean()  # 計算K值的3天移動平均值作為D值
    return data

def calculate_moving_averages(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()   # 計算7日移動平均線
    # data['MA21'] = data['Close'].rolling(window=21).mean()  # 計算21日移動平均線
    return data

def get_df(flag):
    df = pd.read_csv("./2330.TW_data.csv")
    if flag==0:
        df = df[23:]
        return df
    if flag==1:
        df['RSI'] = calculate_rsi(df)
        df = calculate_MACD(df)
        df = calculate_moving_averages(df)
        df = df[23:]
        return df
