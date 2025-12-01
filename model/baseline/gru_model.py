import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, pred_len=1, dropout=0.2):
        """
        GRU 模型（支援多步預測）
        
        Args:
            input_dim: 輸入特徵維度
            hidden_dim: 隱藏層維度
            num_layers: GRU 層數
            output_dim: 每個時間步的輸出維度（通常為 1）
            pred_len: 預測未來幾天
            dropout: Dropout 比例
        """
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 輸出層：從 hidden_dim 映射到 pred_len 個預測值
        self.fc = nn.Linear(hidden_dim, output_dim * pred_len)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            out: (batch_size, pred_len) 如果 pred_len > 1
                 (batch_size,) 如果 pred_len = 1
        """
        # GRU forward
        # out: (batch_size, seq_len, hidden_dim)
        # h_n: (num_layers, batch_size, hidden_dim)
        out, h_n = self.gru(x)
        
        # 取最後一個時間步的輸出
        # last_out: (batch_size, hidden_dim)
        last_out = out[:, -1, :]
        
        # Dropout
        last_out = self.dropout(last_out)
        
        # 全連接層
        # out: (batch_size, output_dim * pred_len)
        out = self.fc(last_out)
        
        if self.pred_len == 1:
            # 單步預測：返回 (batch_size,)
            return out.squeeze(-1)
        else:
            # 多步預測：返回 (batch_size, pred_len)
            return out.view(-1, self.pred_len)