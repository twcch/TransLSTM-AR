import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim=128,        # ✅ 降低複雜度（從 256 降到 128）
        num_layers=2,          # ✅ 降低層數（從 3 降到 2）
        output_dim=1, 
        pred_len=5, 
        dropout=0.2,           # ✅ 降低 dropout（從 0.3 降到 0.2）
        bidirectional=False    # ✅ 移除雙向（過於複雜）
    ):
        """
        簡化版 LSTM 模型
        
        核心理念：
        1. ❌ 不要雙向 LSTM（未來資訊洩漏 + 過度複雜）
        2. ❌ 不要 Attention（LSTM 已有記憶機制）
        3. ✅ 使用 Teacher Forcing（訓練時用真實值）
        4. ✅ 簡單的多步預測（直接輸出 5 個值）
        
        Args:
            input_dim: 輸入特徵維度 (10)
            hidden_dim: LSTM 隱藏層維度 (128)
            num_layers: LSTM 層數 (2)
            output_dim: 輸出維度 (1)
            pred_len: 預測天數 (5)
            dropout: Dropout 率 (0.2)
            bidirectional: 是否雙向（固定 False）
        """
        super(LSTMModel, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ========== 1. Simple LSTM ==========
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # ✅ 單向
        )
        
        # ========== 2. Layer Normalization ==========
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # ========== 3. Simple Output Head ==========
        # 使用一個簡單的 MLP 直接輸出 5 天預測
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * output_dim)
        )
        
        # ========== 4. 初始化權重 ==========
        self._init_weights()
    
    def _init_weights(self):
        """Xavier Uniform 初始化"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # LSTM forget gate bias 初始化為 1
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: [batch, seq_len, input_dim]
               例如 [32, 45, 10]
        
        Returns:
            predictions: [batch, pred_len]
                        例如 [32, 5]
        """
        batch_size = x.size(0)
        
        # ========== Step 1: LSTM ==========
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_dim]
        # h_n: [num_layers, batch, hidden_dim]
        
        # ========== Step 2: 使用最後一個時間步 ==========
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        last_hidden = self.layer_norm(last_hidden)
        
        # ========== Step 3: 直接輸出 5 天預測 ==========
        output = self.output_head(last_hidden)  # [batch, pred_len]
        
        return output


# ========== 測試代碼 ==========
if __name__ == "__main__":
    model = LSTMModel(
        input_dim=10,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        pred_len=5,
        dropout=0.2,
        bidirectional=False
    )
    
    x = torch.randn(32, 45, 10)
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 預期輸出:
    # Input shape:  torch.Size([32, 45, 10])
    # Output shape: torch.Size([32, 5])
    # Model parameters: ~200,000