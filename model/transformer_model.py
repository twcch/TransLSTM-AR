import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 建立一個矩陣 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 使用 sin 和 cos 來編碼位置
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加一個 batch 維度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 註冊為 buffer (不會被視為模型參數更新，但會隨模型保存)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # 將位置編碼加到 Embedding 上
        # 注意：這裡的 x 必須是 batch_first 的格式
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim,  # Encoder 輸入特徵維度
        output_dim,  # Decoder 輸出特徵維度
        model_dim,  # Transformer 內部維度
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout=0.1,
    ):
        super(TransformerSeq2Seq, self).__init__()
        
        # 保存配置參數
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        
        self.model_dim = model_dim

        # --- Encoder 部分 ---
        self.src_input_fc = nn.Linear(input_dim, model_dim)

        # 設定 batch_first=True，讓輸入維持 (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # --- Decoder 部分 ---
        self.tgt_input_fc = nn.Linear(output_dim, model_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # --- 共用組件 ---
        self.pos_encoder = PositionalEncoding(model_dim)
        self.output_fc = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        """
        src: (Batch, Src_Len, input_dim)
        tgt: (Batch, Tgt_Len, output_dim)
        """

        # --- 處理 Device 問題 ---
        # 確保 Mask 與輸入在同一個 Device 上
        if tgt_mask is not None and tgt_mask.device != tgt.device:
            tgt_mask = tgt_mask.to(tgt.device)

        # --- 1. Encoder 流程 ---
        # Input Embedding & Scaling
        src = self.src_input_fc(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        src = self.dropout(src)  # 加入 Dropout

        # Encoder (因為設定了 batch_first=True，不需要 permute)
        memory = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        # --- 2. Decoder 流程 ---
        # Target Embedding & Scaling
        tgt = self.tgt_input_fc(tgt) * math.sqrt(self.model_dim)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)  # 加入 Dropout

        # Decoder
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # --- 3. 輸出映射 ---
        output = self.output_fc(output)

        return output


if __name__ == "__main__":
    # --- 參數設定 ---
    BATCH_SIZE = 32
    SRC_LEN = 20  # 來源序列長度，Encoder 看的歷史長度 (輸入的時間步數)
    TGT_LEN = 10  # 目標序列長度，Decoder 預測的未來長度 (輸出的時間步數)
    INPUT_DIM = 5  # Encoder 輸入特徵維度 e.g. 每天都有 5 個指標: [開盤價, 收盤價, 最高價, 最低價, 交易量]
    OUTPUT_DIM = 1  # Decoder 輸出特徵維度 e.g. 預測未來每天的收盤價
    MODEL_DIM = (
        64  # Transformer 內部運算時的特徵寬度 (這個數字通常要是 NUM_HEADS 的倍數)
    )
    NUM_HEADS = 4  # 多頭注意力機制的頭數 e.g. Head 1 可能專注看「短期波動」, Head 2 可能專注看「長期趨勢」, Head 3 可能專注看「交易量與價格的關係」, 最後再把這 4 個頭看到的結果拼起來
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6

    # 檢查是否有 GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 初始化模型 ---
    model = TransformerSeq2Seq(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=0.1,
    ).to(
        device
    )  # 搬移到 GPU

    # --- 建立假資料 ---
    src = torch.randn(BATCH_SIZE, SRC_LEN, INPUT_DIM).to(device)
    tgt = torch.randn(BATCH_SIZE, TGT_LEN, OUTPUT_DIM).to(device)

    # 建立 Causal Mask (上三角遮罩)
    # 必須確保 mask 也在 GPU 上 (模型 forward 裡有加防呆，但這裡直接放對更好)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(TGT_LEN).to(device)

    # --- Forward Pass ---
    output = model(src, tgt, tgt_mask=tgt_mask)

    print("-" * 30)
    print(f"Source Shape: {src.shape}")
    print(f"Target Shape: {tgt.shape}")
    print(f"Output Shape: {output.shape}")  # 預期: (32, 10, 1)
    print("-" * 30)

    # 簡單驗證 output 是否含有梯度 (代表模型可訓練)
    print(f"Requires Grad: {output.requires_grad}")
