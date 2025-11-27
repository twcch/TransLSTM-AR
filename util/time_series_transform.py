import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class LongFormatDataset(Dataset):
    def __init__(self, df, seq_len=30, pred_len=1, target_col="Close_Scaled"):
        """
        df: 包含 Ticker_ID 和所有 Scaled 欄位的 DataFrame
        seq_len: 回看天數 (Input)
        pred_len: 預測天數 (Output)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 1. 定義特徵欄位
        # 數值特徵 (Open, High, Low, Close, Volume)
        self.feature_cols = [c for c in df.columns if "Scaled" in c]
        # 目標欄位索引 (之後要從 array 中取出來用)
        self.target_idx = self.feature_cols.index(target_col)

        # 2. 核心：依 Ticker_ID 分組，預先切好所有樣本
        # 這樣做比較耗記憶體，但訓練速度最快，邏輯也最簡單
        self.samples = []

        # Groupby 確保我們不會跨公司切分
        for ticker_id, group in df.groupby("Ticker_ID"):
            # 必須按時間排序 (雖然預設通常排好了，但保險起見)
            group = group.sort_values("Date")

            # 取出數值矩陣 (N, Features)
            features_data = group[self.feature_cols].values.astype(np.float32)
            # 取出 ID (雖然這一組都是同個 ID，但為了格式統一)
            ticker_id_val = int(ticker_id)

            # 滑動視窗切分
            num_rows = len(group)
            # 確保長度夠切
            if num_rows <= seq_len + pred_len:
                continue

            for i in range(num_rows - seq_len - pred_len + 1):
                # Input X: 過去 seq_len 天的所有特徵
                x_feat = features_data[i : i + seq_len]

                # Input ID: 這筆資料屬於哪家公司 (由模型 Embedding 用)
                # 我們把它重複 seq_len 次，或者只傳一個都可以，這裡傳一個整數即可
                x_id = ticker_id_val

                # Target Y: 未來 pred_len 天的 Close
                # 這裡示範預測「下一天」
                y = features_data[i + seq_len : i + seq_len + pred_len, self.target_idx]

                self.samples.append((x_feat, x_id, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_feat, x_id, y = self.samples[idx]
        return {
            "x_feat": torch.tensor(x_feat),  # Shape: (Seq_Len, Num_Features)
            "x_id": torch.tensor(x_id, dtype=torch.long),  # Shape: Scalar (整數)
            "y": torch.tensor(y),  # Shape: (Pred_Len)
        }


if __name__ == "__main__":
    train_scaled_data = pd.read_csv("data/processed/train_scaled_data.csv")
    test_scaled_data = pd.read_csv("data/processed/test_scaled_data.csv")

    SEQ_LEN = 30  # 模型回頭看多久的資料
    BATCH_SIZE = 32  # 批次大小

    train_dataset = LongFormatDataset(train_scaled_data, seq_len=SEQ_LEN)
    test_dataset = LongFormatDataset(test_scaled_data, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 測試集通常不需要 shuffle
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 檢查一下吐出來的形狀
    sample = next(iter(train_loader))
    print("數值特徵形狀:", sample["x_feat"].shape)  # (32, 30, 5) -> (Batch, Time, Feat)
    print("公司 ID 形狀:", sample["x_id"].shape)  # (32)       -> (Batch)
    print("預測目標形狀:", sample["y"].shape)  # (32, 1)    -> (Batch, 1)
