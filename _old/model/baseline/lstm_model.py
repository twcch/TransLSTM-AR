import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len, dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.bidirectional = bidirectional
        
        # ✅ 修正：當 num_layers > 1 時才啟用 dropout
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        # 如果是雙向，輸出維度要 *2
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # ✅ 修正：輸出層直接輸出 pred_len 個值（不要額外的維度）
        self.fc = nn.Linear(fc_input_dim, pred_len)
        
        # ✅ 新增：Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        
        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim * num_directions)
        
        # Dropout
        last_output = self.dropout(last_output)
        
        # 全連接層
        out = self.fc(last_output)  # (batch, pred_len)
        
        # ✅ 關鍵：不要 reshape，直接返回 (batch, pred_len)
        return out