import torch
import torch.nn as nn
import math


class TransformerEncoderDecoderModel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=2,
        output_dim=1, 
        pred_len=5, 
        dropout=0.2
    ):
        """
        Encoder-Decoder Transformer for Multi-Step Stock Price Forecasting
        
        Architecture:
        1. Encoder: 處理歷史價格序列 (seq_len 天)
        2. Decoder: 自回歸生成未來預測 (pred_len 天)
        3. Cross-Attention: Decoder 動態關注歷史關鍵時段
        
        Args:
            input_dim: 輸入特徵維度 (10個特徵)
            d_model: Transformer 內部維度 (256)
            nhead: Multi-head attention 的頭數 (8)
            num_encoder_layers: Encoder 層數 (4)
            num_decoder_layers: Decoder 層數 (2)
            output_dim: 輸出維度 (1 = 股價)
            pred_len: 預測天數 (5)
            dropout: Dropout 率 (0.2)
        
        Example:
            >>> model = TransformerEncoderDecoderModel(
            ...     input_dim=10, d_model=256, nhead=8,
            ...     num_encoder_layers=4, num_decoder_layers=2,
            ...     pred_len=5, dropout=0.2
            ... )
            >>> x = torch.randn(32, 60, 10)  # [batch, seq_len, features]
            >>> output = model(x)  # [batch, 5]
        """
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.input_dim = input_dim
        
        # ========== Input Projection Layer ==========
        # 將 input_dim (10) 投影到 d_model (256)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # ========== Positional Encoding ==========
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # ========== Transformer Encoder ==========
        # 處理歷史序列，提取特徵
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # Feed-forward 維度
            dropout=dropout,
            batch_first=True,  # 使用 [batch, seq, feature] 格式
            norm_first=True    # Pre-Layer Normalization (更穩定)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_encoder_layers,
            norm=nn.LayerNorm(d_model)  # 最後的 Layer Norm
        )
        
        # ========== Transformer Decoder ==========
        # 自回歸生成未來預測
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
        
        # ========== Learnable Decoder Input (Query Embeddings) ==========
        # 代表 5 個預測天的「起始向量」
        # 模型會學習這些向量，使其代表「明天」、「後天」...「5天後」
        self.decoder_input = nn.Parameter(torch.randn(pred_len, d_model))
        self._init_decoder_input()
        
        # ========== Output Projection Layer ==========
        # 將 d_model (256) 投影回股價空間 (1)
        self.fc_out = nn.Linear(d_model, output_dim)
        
        # ========== Weight Initialization ==========
        self._init_weights()
    
    def _init_decoder_input(self):
        """初始化 Decoder Input (Query Embeddings)"""
        nn.init.xavier_uniform_(self.decoder_input.unsqueeze(0))
    
    def _init_weights(self):
        """權重初始化 (Xavier Uniform)"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        """
        Forward pass
        
        Args:
            src: [batch, seq_len, input_dim]
                 例如 [32, 60, 10] - 32個樣本，60天歷史，10個特徵
        
        Returns:
            predictions: [batch, pred_len]
                        例如 [32, 5] - 32個樣本，5天預測
        
        Processing Flow:
        1. Input Projection: [batch, 60, 10] → [batch, 60, 256]
        2. Positional Encoding: 加入位置信息
        3. Encoder: 處理歷史 → memory [batch, 60, 256]
        4. Decoder Input: 準備 5 個 queries [batch, 5, 256]
        5. Decoder: 自回歸生成 → [batch, 5, 256]
        6. Output Projection: [batch, 5, 256] → [batch, 5, 1] → [batch, 5]
        """
        batch_size = src.size(0)
        
        # ========== Step 1: Encoder - 處理歷史序列 ==========
        # 1.1 Input Projection
        src = self.input_projection(src)  # [batch, seq_len, d_model]
        
        # 1.2 縮放 (Attention is All You Need 論文建議)
        src = src * math.sqrt(self.d_model)
        
        # 1.3 加入位置編碼
        src = self.pos_encoder(src)
        
        # 1.4 Encoder 處理
        memory = self.encoder(src)  # [batch, seq_len, d_model]
        # memory 包含了歷史序列的所有信息
        
        # ========== Step 2: Decoder - 自回歸生成預測 ==========
        # 2.1 準備 Decoder Input (Query Embeddings)
        tgt = self.decoder_input.unsqueeze(0).expand(batch_size, -1, -1)
        # tgt: [batch, pred_len, d_model]
        # 例如: [32, 5, 256] - 5 個 learnable queries
        
        # 2.2 加入位置編碼 (讓模型知道 Day 1, Day 2, ..., Day 5)
        tgt = self.pos_encoder(tgt)
        
        # 2.3 生成 Causal Mask
        # 確保 Day i 只能看到 Day 1..i-1 的預測
        causal_mask = self._generate_square_subsequent_mask(self.pred_len).to(src.device)
        
        # 🔍 Debug: 列印 mask (只在第一次)
        if not hasattr(self, '_mask_printed'):
            print(f"\n{'='*60}")
            print(f"Decoder Causal Mask (size: {causal_mask.shape}):")
            print(causal_mask)
            print(f"{'='*60}\n")
            self._mask_printed = True
        
        # 2.4 Decoder 處理
        output = self.decoder(
            tgt=tgt,              # [batch, 5, d_model] - 要生成的內容
            memory=memory,        # [batch, 60, d_model] - 歷史信息
            tgt_mask=causal_mask  # [5, 5] - Causal mask
        )
        # output: [batch, pred_len, d_model]
        
        # Decoder 內部發生的事情:
        # - Self-Attention: Day 2 看到 Day 1，Day 3 看到 Day 1-2...
        # - Cross-Attention: 每一天都動態關注歷史序列 (memory)
        # - Feed-Forward: 非線性轉換
        
        # ========== Step 3: Output Projection ==========
        # 3.1 投影到股價空間
        predictions = self.fc_out(output)  # [batch, pred_len, 1]
        
        # 3.2 去掉最後一維
        predictions = predictions.squeeze(-1)  # [batch, pred_len]
        
        return predictions
    
    def _generate_square_subsequent_mask(self, sz):
        """
        生成 Causal Mask (上三角為 -inf，下三角為 0)
        
        這個 mask 確保自回歸特性：
        - Day 1 只能看到自己
        - Day 2 可以看到 Day 1-2
        - Day 5 可以看到 Day 1-5
        
        Args:
            sz: 序列長度 (pred_len = 5)
        
        Returns:
            mask: [sz, sz] 的上三角矩陣
        
        Example for sz=5:
            [[0,    -inf, -inf, -inf, -inf],
             [0,    0,    -inf, -inf, -inf],
             [0,    0,    0,    -inf, -inf],
             [0,    0,    0,    0,    -inf],
             [0,    0,    0,    0,    0   ]]
        
        在 Attention 計算中:
        - 0 的位置: 可以關注 (attention weight 正常計算)
        - -inf 的位置: 不能關注 (attention weight = 0)
        """
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional Encoding (Attention is All You Need 論文)
        
        為序列中的每個位置生成唯一的編碼，讓模型知道：
        - 哪個是「昨天」，哪個是「60天前」
        - 哪個是「明天」，哪個是「5天後」
        
        使用 Sinusoidal Function:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        優點:
        - 可以處理任意長度的序列
        - 相對位置關係清晰
        - 不需要訓練
        
        Args:
            d_model: 模型維度 (256)
            dropout: Dropout 率 (0.1)
            max_len: 最大序列長度 (5000)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 創建位置編碼矩陣 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 計算 div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶數維度使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇數維度使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 維度 [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 註冊為 buffer (不會被當作參數訓練)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x + positional encoding: [batch, seq_len, d_model]
        """
        # 只取需要的長度
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)