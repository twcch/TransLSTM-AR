import sys
import os
from pathlib import Path

# 將專案根目錄加入 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.transformer_model import TransformerSeq2Seq
from util.time_series_transform import LongFormatDataset


class ModelTrainer:
    def __init__(self, model, data, model_type="transformer", batch_size=32, learning_rate=0.0001):
        self.model = model
        self.data = data
        self.model_type = model_type

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # 使用更小的學習率並加入權重衰減
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 添加學習率調度器 - 移除 verbose 參數
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )

        self.loss_fn = nn.MSELoss()
        self.epochs = 100
        self.max_grad_norm = 1.0

    def set_device(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)

    def train(self):
        if self.model_type == "transformer":
            self.train_transformer()
        elif self.model_type == "lstm":
            self.train_lstm()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_transformer(self):
        """訓練 Transformer Seq2Seq 模型"""
        self.model.train()
        
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch in self.dataloader:
                x_feat = batch["x_feat"].float().to(self.device)
                y = batch["y"].float().to(self.device)

                src = x_feat
                batch_size = y.shape[0]

                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)
                if len(y.shape) == 2 and y.shape[-1] == 1:
                    tgt = torch.zeros(batch_size, 1, 1).to(self.device)
                else:
                    tgt = y.unsqueeze(-1)

                tgt_seq_len = tgt.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

                self.optimizer.zero_grad()
                
                try:
                    output = self.model(src, tgt, tgt_mask=tgt_mask)
                    
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"Warning: NaN or Inf detected in output at epoch {epoch+1}")
                        continue
                    
                    if len(y.shape) == 2 and y.shape[-1] == 1:
                        loss = self.loss_fn(output.squeeze(-1), y.squeeze(-1))
                    else:
                        loss = self.loss_fn(output, y.unsqueeze(-1))
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: Invalid loss at epoch {epoch+1}, skipping batch")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                except RuntimeError as e:
                    print(f"Error at epoch {epoch+1}: {e}")
                    continue

            if batch_count == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}]: No valid batches")
                continue
                
            avg_loss = epoch_loss / batch_count
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_model("output/model/best_transformer_model.pth")
            else:
                patience_counter += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}, Best: {best_loss:.6f}")
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print("Training completed!")

    def save_model(self, path):
        """儲存模型 - 確保 dropout 儲存為數值"""
        dropout_value = self.model.dropout
        if isinstance(dropout_value, nn.Dropout):
            dropout_value = dropout_value.p
        elif hasattr(dropout_value, 'p'):
            dropout_value = dropout_value.p
        else:
            dropout_value = 0.1
        
        model_config = {
            'input_dim': self.model.input_dim,
            'output_dim': self.model.output_dim,
            'model_dim': self.model.model_dim,
            'num_heads': self.model.num_heads,
            'num_encoder_layers': self.model.num_encoder_layers,
            'num_decoder_layers': self.model.num_decoder_layers,
            'dropout': float(dropout_value)
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "model_type": self.model_type,
                "model_config": model_config
            },
            path,
        )
        print(f"Model saved to {path}")


if __name__ == "__main__":
    BATCH_SIZE = 32
    SRC_LEN = 20
    TGT_LEN = 10
    INPUT_DIM = 5
    OUTPUT_DIM = 1
    MODEL_DIM = 64
    NUM_HEADS = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    LEARNING_RATE = 0.0001
    
    transformer = TransformerSeq2Seq(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=0.1
    )
    
    train_scaled_data = pd.read_csv("data/processed/train_scaled_data.csv")
    
    train_dataset = LongFormatDataset(
        train_scaled_data, 
        seq_len=SRC_LEN, 
        pred_len=TGT_LEN,
        target_col="Close_Scaled"
    )
    
    trainer = ModelTrainer(
        model=transformer, 
        data=train_dataset,
        model_type="transformer",
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    trainer.set_device("mps")
    trainer.train()
    trainer.save_model("output/model/transformer_model.pth")