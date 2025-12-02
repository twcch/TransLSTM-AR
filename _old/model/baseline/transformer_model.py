import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderDecoderModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        d_model=128,           # ✅ 降低維度（從 256 降到 128）
        nhead=4,               # ✅ 減少頭數（從 8 降到 4）
        num_encoder_layers=3,  # ✅ 減少 Encoder 層數（從 6 降到 3）
        num_decoder_layers=3,  # ✅ 減少 Decoder 層數（從 8 降到 3）
        output_dim=1, 
        pred_len=5, 
        dropout=0.15           # ✅ 適度 dropout（保持 0.15）
    ):
        """
        極簡穩定版 Transformer
        
        策略：
        1. ✅ 降低模型複雜度（小維度、少層數）
        2. ✅ 移除所有複雜機制（無 Multi-Scale、無 Gating）
        3. ✅ 使用最基本的 Decoder Input（只用 Encoder 最後一步）
        4. ✅ 簡單的輸出層（單層 Linear）
        
        Args:
            input_dim: 輸入特徵維度 (10)
            d_model: Transformer 維度 (128)
            nhead: Attention 頭數 (4)
            num_encoder_layers: Encoder 層數 (3)
            num_decoder_layers: Decoder 層數 (3)
            output_dim: 輸出維度 (1)
            pred_len: 預測天數 (5)
            dropout: Dropout 率 (0.15)
        """
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.input_dim = input_dim
        
        # ========== Input Projection ==========
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # ========== Encoder ==========
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # ========== Decoder ==========
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layers,
            num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # ========== 極簡 Decoder Input：只用 Encoder 最後一步 ==========
        # 不使用 Day Embedding，不使用 Gating，完全依賴 Decoder 學習
        
        # ========== 極簡 Output Layer ==========
        # 不使用複雜的 MLP，只用單層 Linear
        self.fc_out = nn.Linear(d_model, output_dim)
        
        # ========== 初始化 ==========
        self._init_weights()
    
    def _init_weights(self):
        """Xavier Uniform 初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, src):
        """
        Forward pass
        
        Args:
            src: [batch, seq_len, input_dim]
                例如 [32, 60, 10]
        
        Returns:
            predictions: [batch, pred_len]
                        例如 [32, 5]
        
        Processing Flow:
        1. Encoder: 提取序列特徵
        2. Decoder Input: 複製 Encoder 最後一步到 pred_len 次
        3. Decoder: 生成預測
        4. Output: 直接輸出
        """
        batch_size = src.size(0)
        
        # ========== Step 1: Encoder ==========
        src = self.input_projection(src)  # [batch, seq_len, d_model]
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        memory = self.encoder(src)  # [batch, seq_len, d_model]
        
        # ========== Step 2: 極簡 Decoder Input ==========
        # 只用 Encoder 最後一個時間步，複製 pred_len 次
        last_hidden = memory[:, -1:, :]  # [batch, 1, d_model]
        tgt = last_hidden.repeat(1, self.pred_len, 1)  # [batch, pred_len, d_model]
        
        # 加入位置編碼
        tgt = self.pos_encoder(tgt)  # [batch, pred_len, d_model]
        
        # ========== Step 3: Decoder ==========
        # Causal mask（防止未來信息洩漏）
        causal_mask = self._generate_square_subsequent_mask(self.pred_len).to(src.device)
        
        output = self.decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=causal_mask
        )  # [batch, pred_len, d_model]
        
        # ========== Step 4: 極簡 Output ==========
        predictions = self.fc_out(output).squeeze(-1)  # [batch, pred_len]
        
        return predictions
    
    def _generate_square_subsequent_mask(self, sz):
        """生成因果遮罩（下三角）"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


# ========== 測試代碼 ==========
if __name__ == "__main__":
    model = TransformerEncoderDecoderModel(
        input_dim=10,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        output_dim=1,
        pred_len=5,
        dropout=0.15
    )
    
    x = torch.randn(32, 60, 10)
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 預期輸出:
    # Input shape:  torch.Size([32, 60, 10])
    # Output shape: torch.Size([32, 5])
    # Model parameters: ~1,500,000 (約 1.5M，比之前的 8M 小很多)