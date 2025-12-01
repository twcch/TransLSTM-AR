import numpy as np


class FeatureEngineering:
    def __init__(self, data):
        self.data = data

    def add_features(self):
        self.data["log_close"] = np.log(self.data["Close"])
        
        self.data["prev_log_close"] = self.data["log_close"].shift(1)
        self.data["prev_high"] = self.data["High"].shift(1)
        self.data["prev_low"] = self.data["Low"].shift(1)
        self.data["prev_open"] = self.data["Open"].shift(1)
        self.data["prev_volume"] = self.data["Volume"].shift(1)
        
        self.data["log_ret"] = self.data["log_close"] - self.data["prev_log_close"]
        self.data["vol_change"] = self.data["prev_volume"].pct_change()
        
        self.data["ma5"] = self.data["log_close"].shift(1).rolling(5).mean()
        self.data["ma20"] = self.data["log_close"].shift(1).rolling(20).mean()
        self.data["ma5_close_ratio"] = self.data["prev_log_close"] / self.data["ma5"]
        self.data["ma20_close_ratio"] = self.data["prev_log_close"] / self.data["ma20"]
        
        self.data["hl_range"] = (self.data["prev_high"] - self.data["prev_low"]) / np.exp(self.data["prev_log_close"])
        
        return self.data

    def drop_na(self):
        self.data = self.data.dropna()
        return self.data