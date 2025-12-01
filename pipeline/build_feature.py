import numpy as np


class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def add_features(self):
        self.data["log_close"] = np.log(self.data["Close"])
        
        # 日 log return (t vs t-1)
        self.data["log_ret"] = np.log(self.data["Close"] / self.data["Close"].shift(1))
        
        # 成交量變化率
        self.data["vol_change"] = self.data["Volume"].pct_change()

        # 移動平均
        self.data["ma5"] = self.data["Close"].rolling(5).mean()
        self.data["ma20"] = self.data["Close"].rolling(20).mean()

        # 收盤價與 MA 的相對比例
        self.data["ma5_close_ratio"] = self.data["Close"] / self.data["ma5"]
        self.data["ma20_close_ratio"] = self.data["Close"] / self.data["ma20"]

        # 高低價振幅
        self.data["hl_range"] = (self.data["High"] - self.data["Low"]) / self.data[
            "Close"
        ]

        return self.data

    def drop_na(self):
        self.data = self.data.dropna()

        return self.data
