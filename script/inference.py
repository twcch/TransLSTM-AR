import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.transformer_model import TransformerSeq2Seq
from util.time_series_transform import LongFormatDataset
import matplotlib.pyplot as plt


class ModelInference:
    def __init__(self, model_path, test_data, seq_len=20, pred_len=10, target_col="Close_Scaled"):
        self.model_path = model_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.device = torch.device("cpu")
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_config' in checkpoint:
            self.model_config = checkpoint['model_config']
            
            # 修正 dropout - 確保轉換為 float
            if 'dropout' in self.model_config:
                dropout = self.model_config['dropout']
                if isinstance(dropout, nn.Dropout):
                    self.model_config['dropout'] = float(dropout.p)
                elif hasattr(dropout, 'p'):
                    self.model_config['dropout'] = float(dropout.p)
                elif not isinstance(dropout, (int, float)):
                    self.model_config['dropout'] = 0.1
                else:
                    self.model_config['dropout'] = float(dropout)
        else:
            print("No model_config found, inferring from state_dict...")
            state_dict = checkpoint['model_state_dict']
            
            num_encoder_layers = 0
            num_decoder_layers = 0
            
            for key in state_dict.keys():
                if 'transformer_encoder.layers.' in key:
                    layer_num = int(key.split('transformer_encoder.layers.')[1].split('.')[0])
                    num_encoder_layers = max(num_encoder_layers, layer_num + 1)
                if 'transformer_decoder.layers.' in key:
                    layer_num = int(key.split('transformer_decoder.layers.')[1].split('.')[0])
                    num_decoder_layers = max(num_decoder_layers, layer_num + 1)
            
            self.model_config = {
                'input_dim': 5,
                'output_dim': 1,
                'model_dim': 64,
                'num_heads': 4,
                'num_encoder_layers': num_encoder_layers if num_encoder_layers > 0 else 3,
                'num_decoder_layers': num_decoder_layers if num_decoder_layers > 0 else 3,
                'dropout': 0.1
            }
        
        print(f"Model config: {self.model_config}")
        
        self.model = TransformerSeq2Seq(**self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
        if isinstance(test_data, pd.DataFrame):
            print("Creating test dataset...")
            self.test_dataset = LongFormatDataset(
                test_data, seq_len=seq_len, pred_len=pred_len, target_col=target_col
            )
        else:
            self.test_dataset = test_data
        
        self.dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        print(f"Test dataset size: {len(self.test_dataset)}")
        
    def set_device(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        print(f"Device set to: {self.device}")
        
    def predict(self):
        print("Starting prediction...")
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                x_feat = batch["x_feat"].float().to(self.device)
                y = batch["y"].float().to(self.device)
                
                batch_size = y.shape[0]
                
                if len(y.shape) == 1:
                    y = y.unsqueeze(-1)
                if len(y.shape) == 2 and y.shape[-1] == 1:
                    tgt = torch.zeros(batch_size, 1, 1).to(self.device)
                else:
                    tgt = y.unsqueeze(-1)
                
                tgt_seq_len = tgt.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
                
                output = self.model(x_feat, tgt, tgt_mask=tgt_mask)
                
                predictions.append(output.squeeze(-1).cpu().numpy())
                targets.append(y.squeeze(-1).cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(self.dataloader)} batches")
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        print("Prediction completed!")
        return predictions, targets
    
    def evaluate(self):
        print("\n" + "="*60)
        print("Starting Model Evaluation")
        print("="*60)
        
        predictions, targets = self.predict()
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        mask = targets != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
        else:
            mape = float('inf')
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print("\nEvaluation Metrics:")
        print("-" * 60)
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²:   {r2_score:.6f}")
        print("-" * 60)
        print(f"Total predictions: {len(predictions)}")
        print("="*60 + "\n")
        
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2_score': r2_score,
            'predictions': predictions, 'targets': targets
        }
    
    def plot_predictions(self, num_samples=200, save_path=None, show_plot=True):
        print(f"\nGenerating prediction plot...")
        predictions, targets = self.predict()
        
        num_samples = min(num_samples, len(predictions))
        
        plt.figure(figsize=(15, 6))
        
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_plot = predictions[:num_samples, 0]
            target_plot = targets[:num_samples, 0]
        else:
            pred_plot = predictions[:num_samples].flatten()
            target_plot = targets[:num_samples].flatten()
        
        plt.plot(target_plot, label='Actual', color='blue', linewidth=2, alpha=0.8)
        plt.plot(pred_plot, label='Predicted', color='red', linewidth=2, alpha=0.7)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Value (Scaled)', fontsize=12)
        plt.title(f'Model Predictions vs Actual Values ({num_samples} samples)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def save_predictions(self, output_path):
        print(f"\nSaving predictions to {output_path}...")
        predictions, targets = self.predict()
        
        # 檢查是否為多步預測
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # 多步預測：展開所有時間步
            num_samples = predictions.shape[0]
            num_steps = predictions.shape[1]
            
            data = {
                'Sample_Index': np.repeat(np.arange(num_samples), num_steps),
                'Time_Step': np.tile(np.arange(num_steps), num_samples),
                'Actual': targets.flatten(),
                'Predicted': predictions.flatten(),
                'Error': (predictions - targets).flatten(),
                'Abs_Error': np.abs(predictions - targets).flatten()
            }
        else:
            # 單步預測
            data = {
                'Sample_Index': np.arange(len(predictions)),
                'Actual': targets.flatten(),
                'Predicted': predictions.flatten(),
                'Error': (predictions - targets).flatten(),
                'Abs_Error': np.abs(predictions - targets).flatten()
            }
        
        predictions_df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved! ({len(predictions_df)} rows)")
        print(f"Prediction shape: {predictions.shape}, Target shape: {targets.shape}")

    def save_evaluation_metrics(self, results, output_path):
        """保存評估指標到 txt 文件"""
        print(f"\nSaving evaluation metrics to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Model Evaluation Results\n")
            f.write("="*60 + "\n\n")
            
            f.write("Model Configuration:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Sequence Length: {self.seq_len}\n")
            f.write(f"Prediction Length: {self.pred_len}\n")
            f.write(f"Target Column: {self.target_col}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Model Architecture:\n")
            f.write("-" * 60 + "\n")
            for key, value in self.model_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("Evaluation Metrics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"MSE (Mean Squared Error):      {results['mse']:.6f}\n")
            f.write(f"RMSE (Root Mean Squared Error): {results['rmse']:.6f}\n")
            f.write(f"MAE (Mean Absolute Error):      {results['mae']:.6f}\n")
            f.write(f"MAPE (Mean Absolute % Error):   {results['mape']:.2f}%\n")
            f.write(f"R² Score:                       {results['r2_score']:.6f}\n")
            f.write("-" * 60 + "\n\n")
            
            f.write("Data Summary:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Predictions: {len(results['predictions'])}\n")
            f.write(f"Prediction Shape: {results['predictions'].shape}\n")
            f.write(f"Target Shape: {results['targets'].shape}\n")
            f.write("="*60 + "\n")
        
        print(f"Evaluation metrics saved successfully!")

if __name__ == "__main__":
    MODEL_PATH = "output/model/best_transformer_model.pth"
    TEST_DATA_PATH = "data/processed/test_scaled_data.csv"
    OUTPUT_DIR = "output"
    
    SEQ_LEN = 20
    PRED_LEN = 10
    TARGET_COL = "Close_Scaled"
    
    print("Loading test data...")
    test_scaled_data = pd.read_csv(TEST_DATA_PATH)
    print(f"Test data loaded: {len(test_scaled_data)} rows")
    
    inference = ModelInference(
        model_path=MODEL_PATH,
        test_data=test_scaled_data,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        target_col=TARGET_COL
    )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    inference.set_device(device)
    
    results = inference.evaluate()
    inference.plot_predictions(num_samples=200, save_path=f"{OUTPUT_DIR}/predictions.png", show_plot=False)
    inference.save_predictions(f"{OUTPUT_DIR}/predictions.csv")
    
    results = inference.evaluate()
    inference.save_evaluation_metrics(results, f"{OUTPUT_DIR}/evaluation_metrics.txt")
    inference.plot_predictions(num_samples=200, save_path=f"{OUTPUT_DIR}/predictions.png", show_plot=False)
    inference.save_predictions(f"{OUTPUT_DIR}/predictions.csv")
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)