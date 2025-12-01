import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_len=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 新增: LayerNorm 穩定長期預測
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 新增: 額外的全連接層增加表達能力
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim * pred_len)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        
        # 取最後時間步並應用 LayerNorm
        out = out[:, -1, :]
        out = self.layer_norm(out)
        
        # 通過額外的全連接層
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        if self.pred_len > 1:
            return out.view(-1, self.pred_len)
        return out.squeeze(-1)