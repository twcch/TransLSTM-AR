import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim=64,         # ✅ 大幅降低（從 128 降到 64）
        num_layers=1,          # ✅ 只用 1 層（LSTM 容易梯度問題）
        output_dim=1, 
        pred_len=5, 
        dropout=0.0,           # ✅ 移除 dropout（單層不需要）
        bidirectional=False
    ):
        """
        極簡版 LSTM - 解決梯度問題
        
        核心策略：
        1. ✅ 降低複雜度（單層 + 小 hidden_dim）
        2. ✅ 移除 dropout（避免訓練不穩定）
        3. ✅ Layer Normalization（穩定梯度）
        4. ✅ Residual Connection（緩解梯度消失）
        """
        super(LSTMModel, self).__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ========== Input Layer Norm ==========
        self.input_norm = nn.LayerNorm(input_dim)
        
        # ========== LSTM ==========
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.0  # 單層不用 dropout
        )
        
        # ========== Layer Norm (穩定 LSTM 輸出) ==========
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # ========== Output Head (極簡設計) ==========
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, pred_len * output_dim)
        )
        
        # ========== Residual Projection (選用) ==========
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None
        
        self._init_weights()
    
    def _init_weights(self):
        """保守的權重初始化（避免梯度爆炸）"""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data, gain=0.5)  # ✅ gain < 1
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data, gain=0.5)      # ✅ gain < 1
                elif 'bias' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)  # forget gate bias = 1
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        
        Returns:
            [batch, pred_len]
        """
        batch_size = x.size(0)
        
        # ========== Input Normalization ==========
        x = self.input_norm(x)
        
        # ========== LSTM ==========
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_dim]
        
        # ========== 使用最後時間步 ==========
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
        last_hidden = self.layer_norm(last_hidden)
        
        # ========== Residual Connection (選用) ==========
        if self.residual_proj is not None:
            residual = self.residual_proj(x[:, -1, :])
            last_hidden = last_hidden + residual
        
        # ========== Output ==========
        output = self.output_head(last_hidden)  # [batch, pred_len]
        
        return output


# ========== 測試 ==========
if __name__ == "__main__":
    model = LSTMModel(
        input_dim=10,
        hidden_dim=64,
        num_layers=1,
        pred_len=5,
    )
    
    x = torch.randn(32, 45, 10)
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")