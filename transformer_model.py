import math
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # 預先計算 Position Encoding
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)

        # 使用 log 空間計算 division_term 以提高數值穩定性
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )

        # PE(pos, 2i) = sin(...)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(...)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # 調整維度以配合 batch_first=True -> (1, max_len, dim_model)
        pos_encoding = pos_encoding.unsqueeze(0)

        # 註冊為 buffer (不參與梯度更新，但會隨模型保存)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        # token_embedding size: (Batch_Size, Seq_Len, Dim_Model)
        # 這裡會自動廣播 (Broadcasting)
        seq_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_encoding[:, :seq_len, :])


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # 1. Embedding 層
        self.embedding = nn.Embedding(num_tokens, dim_model)

        # 2. Positional Encoding
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        # 3. Transformer 主體 (啟用 batch_first=True)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True,  # 重要改動：輸入輸出皆為 (Batch, Seq, Feature)
        )

        # 4. 輸出層
        self.out = nn.Linear(dim_model, num_tokens)

    def get_tgt_mask(self, size) -> torch.Tensor:
        """
        生成因果遮罩 (Causal Mask)，用於遮蔽未來資訊。
        回傳一個上三角矩陣，對角線與下方為 0，上方為 -inf。
        """
        mask = torch.tril(torch.ones(size, size) == 1)  # 下三角為 True
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # 上三角填入 -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # 下三角填入 0
        return mask

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # src: (Batch_Size, Source_Seq_Len)
        # tgt: (Batch_Size, Target_Seq_Len)

        # 1. 生成 Target Mask (Causal Mask) - 強制 Decoder 不能看未來
        # device 需與輸入資料一致
        tgt_mask = self.get_tgt_mask(tgt.size(1)).to(src.device)

        # 2. Embedding + Positional Encoding
        # 注意：乘以 sqrt(dim_model) 是論文中的細節，保持數值分佈
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # 3. Transformer Blocks
        # 這裡傳入 masks
        transformer_out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,  # 防止偷看未來
            src_key_padding_mask=src_padding_mask,  # 忽略 padding token
            tgt_key_padding_mask=tgt_padding_mask,  # 忽略 padding token
        )

        # 4. Final Linear Layer
        out = self.out(transformer_out)

        return out


# --- 測試與驗證代碼 (Example Usage) ---

if __name__ == "__main__":
    # 設定參數
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tokens = 1000  # 詞彙表大小
    dim_model = 512  # Embedding 維度
    num_heads = 8  # Attention Heads
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout_p = 0.1

    # 實例化模型
    model = Transformer(
        num_tokens=num_tokens,
        dim_model=dim_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout_p=dropout_p,
    ).to(device)

    # 模擬輸入資料 (Batch Size=2, Seq Len=10)
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 10

    src_data = torch.randint(0, num_tokens, (batch_size, src_seq_len)).to(device)
    tgt_data = torch.randint(0, num_tokens, (batch_size, tgt_seq_len)).to(device)

    # 模擬 Padding Mask (假設最後幾個 token 是 padding)
    # True 代表該位置是 padding，需要被忽略
    src_pad_mask = torch.zeros((batch_size, src_seq_len)).bool().to(device)
    src_pad_mask[0, -2:] = True  # 第一筆資料最後兩個是 padding

    # 前向傳播
    output = model(src=src_data, tgt=tgt_data, src_padding_mask=src_pad_mask)

    print(f"Input shape: {src_data.shape}")
    print(f"Output shape: {output.shape}")  # 預期: (2, 10, 1000)
    print("模型運作成功！")
