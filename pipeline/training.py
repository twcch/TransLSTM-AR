import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy
from matplotlib.dates import DateFormatter, MonthLocator


class Training:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        learning_rate=0.001,
        epochs=50,
        device=None,
        early_stopping_patience=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        if device is None:
            self.device = self.get_device()
        else:
            self.device = device

    def get_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device

    def train_model(self, model, train_loader, val_loader, model_name="Model"):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # 學習率調度器 - 針對不同模型使用不同策略
        if "LSTM" in model_name:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.3,      # LSTM 使用更激進的衰減
                patience=10,
                verbose=True,
                min_lr=1e-7,
                threshold=1e-4,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=15,
                verbose=True,
                min_lr=1e-7,
                threshold=1e-4,
            )

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state = None
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        print(f"Early Stopping Patience: {self.early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            # ========== Train Phase ==========
            model.train()
            total_train_loss = 0.0
            n_train = 0
            
            # 梯度監控
            total_grad_norm = 0.0
            grad_count = 0

            for batch_data in train_loader:
                if len(batch_data) == 3:
                    X_batch, y_batch, tid_batch = batch_data
                else:
                    X_batch, y_batch, tid_batch, _ = batch_data
                
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()

                # 梯度裁剪 + 監控
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_grad_norm += grad_norm.item()
                grad_count += 1

                optimizer.step()

                total_train_loss += loss.item() * X_batch.size(0)
                n_train += X_batch.size(0)

            avg_train_loss = total_train_loss / n_train
            avg_grad_norm = total_grad_norm / grad_count if grad_count > 0 else 0.0

            # ========== Validation Phase ==========
            model.eval()
            total_val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3:
                        X_batch, y_batch, tid_batch = batch_data
                    else:
                        X_batch, y_batch, tid_batch, _ = batch_data
                    
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    total_val_loss += loss.item() * X_batch.size(0)
                    n_val += X_batch.size(0)

            avg_val_loss = total_val_loss / n_val

            # 更新學習率調度器
            scheduler.step(avg_val_loss)

            # ========== Early Stopping Logic ==========
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
                marker = " ⭐"
            else:
                patience_counter += 1
                marker = ""

            # 顯示訓練進度（包含梯度信息）
            if epoch % 10 == 0 or epoch == 1 or marker:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"[{model_name}] Epoch {epoch:03d} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {avg_val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Grad: {avg_grad_norm:.4f}{marker}"
                )

            # 檢查是否需要提前停止
            if (
                self.early_stopping_patience is not None
                and patience_counter >= self.early_stopping_patience
            ):
                print(
                    f"\n[{model_name}] Early stopping triggered at epoch {epoch}!"
                )
                print(
                    f"[{model_name}] Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}"
                )
                break

        # 恢復最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(
                f"\n[{model_name}] ✓ Restored best model from epoch {best_epoch} "
                f"(Val Loss: {best_val_loss:.6f})"
            )
        
        print(f"{'='*60}\n")
        return best_epoch

    def calculate_r2_score(self, y_true, y_pred):
        """
        計算 R² (coefficient of determination)
        
        R² = 1 - (SS_res / SS_tot)
        where:
            SS_res = Σ(y_true - y_pred)²  (殘差平方和)
            SS_tot = Σ(y_true - y_mean)²  (總平方和)
        
        R² = 1 表示完美預測
        R² = 0 表示預測等同於使用平均值
        R² < 0 表示預測比平均值還差
        """
        # 轉換為 numpy array
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        # 確保是 1D array
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 計算 R²
        ss_res = np.sum((y_true - y_pred) ** 2)  # 殘差平方和
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 總平方和
        
        if ss_tot == 0:
            return 0.0  # 避免除以零
        
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def evaluate_model(
        self, model, test_loader, feature_cols, scalers, id2ticker, model_name="Model", pred_len=1
    ):
        """評估模型並顯示所有股票的結果（支援多步預測、日期軸和修正的 R² 指標）"""
        model.eval()
        criterion = nn.MSELoss()

        total_test_loss_scaled = 0.0
        n_test = 0

        all_preds_scaled = []
        all_targets_scaled = []
        all_ticker_ids = []
        all_dates = []

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on Test Set")
        print(f"Prediction Horizon: {pred_len} day{'s' if pred_len > 1 else ''}")
        print(f"{'='*60}\n")

        # 檢查是否有日期資訊
        has_dates = False
        first_batch = next(iter(test_loader))
        if len(first_batch) == 4:
            has_dates = True
            print(f"✓ Date information detected - will use date axis in plots")
        else:
            print(f"✗ No date information - will use time step index in plots")
        print()

        # ========== Test Phase ==========
        with torch.no_grad():
            for batch_data in test_loader:
                if has_dates:
                    X_batch, y_batch, tid_batch, date_batch = batch_data
                    all_dates.extend(date_batch)
                else:
                    X_batch, y_batch, tid_batch = batch_data
                
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = model(X_batch)

                loss = criterion(preds, y_batch)
                total_test_loss_scaled += loss.item() * X_batch.size(0)
                n_test += X_batch.size(0)

                all_preds_scaled.append(preds.cpu().numpy())
                all_targets_scaled.append(y_batch.cpu().numpy())
                all_ticker_ids.append(tid_batch.numpy())

        avg_test_loss_scaled = total_test_loss_scaled / n_test

        all_preds_scaled = np.concatenate(all_preds_scaled)
        all_targets_scaled = np.concatenate(all_targets_scaled)
        all_ticker_ids = np.concatenate(all_ticker_ids)

        # ========== 反轉標準化 + 反 log → 真實股價 ==========
        preds_price_list = []
        targets_price_list = []
        ticker_names = []

        D = len(feature_cols)

        for s_pred, s_true, tid in zip(
            all_preds_scaled, all_targets_scaled, all_ticker_ids
        ):
            ticker = id2ticker[int(tid)]
            scaler = scalers[ticker]
            ticker_names.append(ticker)

            if pred_len == 1:
                fake_pred = np.zeros((1, D), dtype=np.float32)
                fake_true = np.zeros((1, D), dtype=np.float32)
                fake_pred[0, 0] = s_pred
                fake_true[0, 0] = s_true

                inv_pred = scaler.inverse_transform(fake_pred)[0, 0]
                inv_true = scaler.inverse_transform(fake_true)[0, 0]

                price_pred = math.exp(inv_pred)
                price_true = math.exp(inv_true)

                preds_price_list.append(price_pred)
                targets_price_list.append(price_true)
            else:
                price_preds = []
                price_trues = []
                
                for day_idx in range(pred_len):
                    fake_pred = np.zeros((1, D), dtype=np.float32)
                    fake_true = np.zeros((1, D), dtype=np.float32)
                    fake_pred[0, 0] = s_pred[day_idx]
                    fake_true[0, 0] = s_true[day_idx]

                    inv_pred = scaler.inverse_transform(fake_pred)[0, 0]
                    inv_true = scaler.inverse_transform(fake_true)[0, 0]

                    price_pred = math.exp(inv_pred)
                    price_true = math.exp(inv_true)

                    price_preds.append(price_pred)
                    price_trues.append(price_true)
                
                preds_price_list.append(price_preds)
                targets_price_list.append(price_trues)

        preds_price = np.array(preds_price_list)
        targets_price = np.array(targets_price_list)

        # ========== 計算整體評估指標（修正 R² 計算）==========
        mse_price = np.mean((preds_price - targets_price) ** 2)
        rmse_price = math.sqrt(mse_price)
        mae_price = np.mean(np.abs(preds_price - targets_price))
        mape_price = np.mean(np.abs((targets_price - preds_price) / targets_price)) * 100
        
        # ✅ 修正: 分股票計算 R² 後加權平均
        r2_per_stock = []
        stock_counts = []
        
        for ticker in sorted(set(ticker_names)):
            ticker_mask = np.array(ticker_names) == ticker
            ticker_preds = preds_price[ticker_mask]
            ticker_targets = targets_price[ticker_mask]
            
            ticker_r2 = self.calculate_r2_score(ticker_targets, ticker_preds)
            r2_per_stock.append(ticker_r2)
            stock_counts.append(len(ticker_preds))
        
        # 加權平均 R² (正確的 Overall R²)
        total_samples = sum(stock_counts)
        r2_weighted = sum(r2 * count for r2, count in zip(r2_per_stock, stock_counts)) / total_samples
        
        # 混合計算的 R² (可能虛高)
        r2_mixed = self.calculate_r2_score(targets_price, preds_price)

        print(f"{'='*60}")
        print(f"Overall Test Results ({model_name})")
        print(f"{'='*60}")
        print(f"Test MSE (scaled log_close):  {avg_test_loss_scaled:.6f}")
        print(f"Test RMSE (scaled log_close): {math.sqrt(avg_test_loss_scaled):.6f}")
        print(f"\n{'─'*60}")
        print(f"Test MSE (real price):        {mse_price:.4f}")
        print(f"Test RMSE (real price):       {rmse_price:.4f}")
        print(f"Test MAE (real price):        {mae_price:.4f}")
        print(f"Test MAPE (real price):       {mape_price:.2f}%")
        print(f"\n{'─'*60}")
        print(f"R² Metrics:")
        print(f"  Weighted Average R²:        {r2_weighted:.4f}  ✓ Correct")
        print(f"  Mixed Stocks R²:            {r2_mixed:.4f}  ⚠️  May be inflated")
        
        # 警告虛高的 R²
        if abs(r2_mixed - r2_weighted) > 0.1:
            print(f"\n⚠️  WARNING: Large difference between mixed and weighted R²!")
            print(f"   Difference: {abs(r2_mixed - r2_weighted):.4f}")
            print(f"   This indicates stocks have very different price ranges.")
            print(f"   Use 'Weighted Average R²' as the correct metric.")
        
        if pred_len > 1:
            print(f"\n{'─'*60}")
            print(f"Per-Day Average MAPE & R²:")
            for day in range(pred_len):
                if preds_price.ndim == 2:
                    day_mape = np.mean(np.abs((targets_price[:, day] - preds_price[:, day]) / targets_price[:, day])) * 100
                    day_r2 = self.calculate_r2_score(targets_price[:, day], preds_price[:, day])
                    print(f"  Day {day+1}: MAPE={day_mape:.2f}%, R²={day_r2:.4f}")
        
        print(f"{'='*60}\n")

        # ========== 分股票評估（包含 R²）==========
        print(f"{'='*60}")
        print(f"Per-Stock Test Results ({model_name})")
        print(f"{'='*60}\n")

        for ticker in sorted(set(ticker_names)):
            ticker_mask = np.array(ticker_names) == ticker
            ticker_preds = preds_price[ticker_mask]
            ticker_targets = targets_price[ticker_mask]

            ticker_mse = np.mean((ticker_preds - ticker_targets) ** 2)
            ticker_rmse = math.sqrt(ticker_mse)
            ticker_mae = np.mean(np.abs(ticker_preds - ticker_targets))
            ticker_mape = np.mean(np.abs((ticker_targets - ticker_preds) / ticker_targets)) * 100
            ticker_r2 = self.calculate_r2_score(ticker_targets, ticker_preds)

            print(f"[{ticker}]")
            print(f"  Samples: {len(ticker_preds)}")
            print(f"  MSE:     {ticker_mse:.4f}")
            print(f"  RMSE:    {ticker_rmse:.4f}")
            print(f"  MAE:     {ticker_mae:.4f}")
            print(f"  MAPE:    {ticker_mape:.2f}%")
            print(f"  R²:      {ticker_r2:.4f}")
            
            if pred_len > 1 and ticker_preds.ndim == 2:
                print(f"  Per-Day Metrics:")
                for day in range(pred_len):
                    day_mape = np.mean(np.abs((ticker_targets[:, day] - ticker_preds[:, day]) / ticker_targets[:, day])) * 100
                    day_r2 = self.calculate_r2_score(ticker_targets[:, day], ticker_preds[:, day])
                    print(f"    Day {day+1}: MAPE={day_mape:.2f}%, R²={day_r2:.4f}")
            print()

        # ========== 繪製各股票預測圖 ==========
        print(f"{'='*60}")
        print(f"Generating Prediction Plots ({model_name})")
        print(f"{'='*60}\n")

        for ticker in sorted(set(ticker_names)):
            ticker_id = None
            for tid, tname in id2ticker.items():
                if tname == ticker:
                    ticker_id = tid
                    break

            if ticker_id is not None:
                mask = all_ticker_ids == ticker_id
                idx = np.where(mask)[0]

                if len(idx) > 0:
                    n_plot = min(200, len(idx))
                    sel_idx = idx[:n_plot]

                    # 準備 x 軸資料
                    if has_dates:
                        dates_for_ticker = [all_dates[i] for i in sel_idx]
                        x_axis = pd.to_datetime(dates_for_ticker)
                        x_label = "Date"
                        use_date_format = True
                    else:
                        x_axis = np.arange(len(sel_idx))
                        x_label = "Time Step"
                        use_date_format = False

                    if pred_len == 1:
                        # 單步預測圖
                        fig = plt.figure(figsize=(16, 6))
                        
                        ax1 = plt.subplot(1, 2, 1)
                        plt.plot(
                            x_axis,
                            targets_price[sel_idx],
                            label=f"{ticker} True Price",
                            linewidth=2.5,
                            alpha=0.9,
                            color='#2E86AB',
                            marker='o',
                            markersize=4,
                        )
                        plt.plot(
                            x_axis,
                            preds_price[sel_idx],
                            label=f"{ticker} Predicted ({model_name})",
                            linewidth=2,
                            alpha=0.8,
                            color='#A23B72',
                            marker='x',
                            markersize=4,
                        )
                        
                        if use_date_format:
                            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                            ax1.xaxis.set_major_locator(MonthLocator(interval=2))
                            plt.xticks(rotation=45, ha='right')
                        
                        # 加入 R² 到圖例
                        ticker_mask = np.array(ticker_names) == ticker
                        ticker_r2 = self.calculate_r2_score(targets_price[ticker_mask], preds_price[ticker_mask])
                        
                        plt.legend(loc='best', fontsize=10, title=f"R² = {ticker_r2:.4f}")
                        plt.title(f"{ticker} - Prediction vs True Price ({model_name})", fontsize=12, fontweight='bold')
                        plt.xlabel(x_label, fontsize=10)
                        plt.ylabel("Price", fontsize=10)
                        plt.grid(True, alpha=0.3, linestyle='--')

                        ax2 = plt.subplot(1, 2, 2)
                        errors = preds_price[sel_idx] - targets_price[sel_idx]
                        plt.plot(x_axis, errors, linewidth=2, color='#F18F01', alpha=0.8)
                        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
                        
                        if use_date_format:
                            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                            ax2.xaxis.set_major_locator(MonthLocator(interval=2))
                            plt.xticks(rotation=45, ha='right')
                        
                        plt.title(f"{ticker} - Prediction Errors ({model_name})", fontsize=12, fontweight='bold')
                        plt.xlabel(x_label, fontsize=10)
                        plt.ylabel("Error (Pred - True)", fontsize=10)
                        plt.grid(True, alpha=0.3, linestyle='--')

                        plt.tight_layout()

                    else:
                        # 多步預測圖
                        fig = plt.figure(figsize=(18, 12))
                        
                        ax1 = plt.subplot(2, 1, 1)
                        
                        plt.plot(
                            x_axis,
                            targets_price[sel_idx, 0],
                            label=f"{ticker} True Price (Day 1)",
                            linewidth=3,
                            alpha=0.9,
                            color='black',
                            marker='o',
                            markersize=4,
                            zorder=10,
                        )
                        
                        colors = plt.cm.viridis(np.linspace(0, 1, pred_len))
                        for day in range(pred_len):
                            plt.plot(
                                x_axis,
                                preds_price[sel_idx, day],
                                label=f"Predicted Day {day+1}",
                                linewidth=2,
                                alpha=0.7,
                                color=colors[day],
                                marker='x',
                                markersize=3,
                            )
                        
                        if use_date_format:
                            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                            ax1.xaxis.set_major_locator(MonthLocator(interval=2))
                            plt.xticks(rotation=45, ha='right')
                        
                        # 加入整體 R²
                        ticker_mask = np.array(ticker_names) == ticker
                        ticker_r2 = self.calculate_r2_score(targets_price[ticker_mask], preds_price[ticker_mask])
                        
                        plt.legend(loc='best', fontsize=9, ncol=2, title=f"Overall R² = {ticker_r2:.4f}")
                        plt.title(f"{ticker} - Multi-Step Predictions ({model_name}, {pred_len} days)", 
                                 fontsize=13, fontweight='bold')
                        plt.xlabel(x_label, fontsize=11)
                        plt.ylabel("Price", fontsize=11)
                        plt.grid(True, alpha=0.3, linestyle='--')

                        ax2 = plt.subplot(2, 1, 2)
                        day_mapes = []
                        day_r2s = []
                        for day in range(pred_len):
                            day_mape = np.mean(np.abs((targets_price[sel_idx, day] - preds_price[sel_idx, day]) / targets_price[sel_idx, day])) * 100
                            day_r2 = self.calculate_r2_score(targets_price[sel_idx, day], preds_price[sel_idx, day])
                            day_mapes.append(day_mape)
                            day_r2s.append(day_r2)
                        
                        x_pos = np.arange(1, pred_len + 1)
                        width = 0.35
                        
                        ax2_twin = ax2.twinx()
                        
                        bars1 = ax2.bar(x_pos - width/2, day_mapes, width, color=colors, alpha=0.8, label='MAPE', edgecolor='black')
                        bars2 = ax2_twin.bar(x_pos + width/2, day_r2s, width, color='gray', alpha=0.6, label='R²', edgecolor='black')
                        
                        for bar, mape_val in zip(bars1, day_mapes):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{mape_val:.1f}%',
                                    ha='center', va='bottom', fontsize=8)
                        
                        for bar, r2_val in zip(bars2, day_r2s):
                            height = bar.get_height()
                            ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                                         f'{r2_val:.3f}',
                                         ha='center', va='bottom', fontsize=8)
                        
                        ax2.set_xlabel("Prediction Day", fontsize=11)
                        ax2.set_ylabel("MAPE (%)", fontsize=11, color='blue')
                        ax2_twin.set_ylabel("R²", fontsize=11, color='red')
                        ax2.set_title(f"{ticker} - MAPE & R² by Prediction Day", fontsize=12, fontweight='bold')
                        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
                        ax2.set_xticks(x_pos)
                        
                        ax2.legend(loc='upper left')
                        ax2_twin.legend(loc='upper right')
                        
                        plt.tight_layout()

                    # 儲存圖片
                    plot_path = f"output/picture/{model_name.lower()}_{ticker.replace('.', '_')}_prediction.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    print(f"[{ticker}] ✓ Plot saved to {plot_path}")
                    plt.close()
                else:
                    print(f"[{ticker}] ✗ No test samples found")

        print(f"\n{'='*60}\n")

        # ========== 返回評估結果字典（使用正確的 R²）==========
        results = {
            "overall": {
                "mse": float(mse_price),
                "rmse": float(rmse_price),
                "mae": float(mae_price),
                "mape": float(mape_price),
                "r2_weighted": float(r2_weighted),  # 正確的 R²
                "r2_mixed": float(r2_mixed),        # 參考用
            },
            "per_stock": {},
        }

        for ticker in sorted(set(ticker_names)):
            ticker_mask = np.array(ticker_names) == ticker
            ticker_preds = preds_price[ticker_mask]
            ticker_targets = targets_price[ticker_mask]

            ticker_results = {
                "mse": float(np.mean((ticker_preds - ticker_targets) ** 2)),
                "rmse": float(math.sqrt(np.mean((ticker_preds - ticker_targets) ** 2))),
                "mae": float(np.mean(np.abs(ticker_preds - ticker_targets))),
                "mape": float(np.mean(np.abs((ticker_targets - ticker_preds) / ticker_targets)) * 100),
                "r2": float(self.calculate_r2_score(ticker_targets, ticker_preds)),
            }
            
            if pred_len > 1 and ticker_preds.ndim == 2:
                ticker_results["per_day_mape"] = {}
                ticker_results["per_day_r2"] = {}
                for day in range(pred_len):
                    day_mape = np.mean(np.abs((ticker_targets[:, day] - ticker_preds[:, day]) / ticker_targets[:, day])) * 100
                    day_r2 = self.calculate_r2_score(ticker_targets[:, day], ticker_preds[:, day])
                    ticker_results["per_day_mape"][f"day_{day+1}"] = float(day_mape)
                    ticker_results["per_day_r2"][f"day_{day+1}"] = float(day_r2)
            
            results["per_stock"][ticker] = ticker_results

        return results