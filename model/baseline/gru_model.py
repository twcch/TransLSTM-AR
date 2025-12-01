import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len, dropout=0.0):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        # ✅ 修正：當 num_layers > 1 時才啟用 dropout
        gru_dropout = dropout if num_layers > 1 else 0.0
        
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        
        # ✅ 修正：輸出層直接輸出 pred_len 個值
        self.fc = nn.Linear(hidden_dim, pred_len)
        
        # ✅ 新增：Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # GRU forward
        gru_out, h_n = self.gru(x)
        # gru_out: (batch, seq_len, hidden_dim)
        
        # 取最後一個時間步的輸出
        last_output = gru_out[:, -1, :]  # (batch, hidden_dim)
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全連接層
        out = self.fc(last_output)  # (batch, pred_len)
        
        # ✅ 關鍵：不要 reshape，直接返回 (batch, pred_len)
        return out