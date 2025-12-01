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
        """訓練模型"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # 學習率調度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=15,
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
            scheduler.step(avg_val_loss)

            # ========== Early Stopping ==========
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
                marker = " ⭐"
            else:
                patience_counter += 1
                marker = ""

            if epoch % 10 == 0 or epoch == 1 or marker:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"[{model_name}] Epoch {epoch:03d} | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {avg_val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Grad: {avg_grad_norm:.4f}{marker}"
                )

            if (
                self.early_stopping_patience is not None
                and patience_counter >= self.early_stopping_patience
            ):
                print(f"\n[{model_name}] Early stopping at epoch {epoch}!")
                print(f"[{model_name}] Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
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
        """計算 R²"""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def evaluate_model(
        self, model, test_loader, feature_cols, scalers, id2ticker, model_name="Model", pred_len=1
    ):
        """靜態預測評估（一次性預測所有天數）"""
        model.eval()
        criterion = nn.MSELoss()

        total_test_loss_scaled = 0.0
        n_test = 0

        all_preds_scaled = []
        all_targets_scaled = []
        all_ticker_ids = []
        all_dates = []

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on Test Set (STATIC Mode)")
        print(f"Prediction Horizon: {pred_len} day{'s' if pred_len > 1 else ''}")
        print(f"{'='*60}\n")

        # 檢查日期
        has_dates = False
        first_batch = next(iter(test_loader))
        if len(first_batch) == 4:
            has_dates = True

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

        # ========== 反標準化 ==========
        preds_price, targets_price, ticker_names = self._inverse_transform(
            all_preds_scaled, all_targets_scaled, all_ticker_ids,
            feature_cols, scalers, id2ticker, pred_len
        )

        # ========== 計算指標 ==========
        results = self._calculate_metrics(
            preds_price, targets_price, ticker_names, 
            avg_test_loss_scaled, model_name, pred_len
        )

        # ========== 繪圖 ==========
        self._plot_predictions(
            preds_price, targets_price, ticker_names, all_ticker_ids,
            id2ticker, all_dates, has_dates, model_name, pred_len, mode="static"
        )

        return results

    def evaluate_model_autoregressive(
        self, model, test_loader, feature_cols, scalers, id2ticker, model_name="Model", pred_len=5
    ):
        """
        自回歸預測評估（滾動預測）
        
        每次只預測 Day 1，然後將預測結果加入輸入序列，再預測 Day 2，依此類推
        """
        model.eval()

        all_preds_rolling = []
        all_targets = []
        all_ticker_ids = []
        all_dates = []

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on Test Set (AUTOREGRESSIVE Mode)")
        print(f"Prediction Horizon: {pred_len} days")
        print(f"{'='*60}\n")

        # 檢查日期
        has_dates = False
        first_batch = next(iter(test_loader))
        if len(first_batch) == 4:
            has_dates = True

        with torch.no_grad():
            for batch_data in test_loader:
                if has_dates:
                    X_batch, y_batch, tid_batch, date_batch = batch_data
                    all_dates.extend(date_batch)
                else:
                    X_batch, y_batch, tid_batch = batch_data

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                batch_size, seq_len, n_features = X_batch.shape

                # ========== 滾動預測 ==========
                current_input = X_batch.clone()
                rolling_preds = []

                for day in range(pred_len):
                    # 預測下一天
                    pred = model(current_input)  # [batch, pred_len] or [batch, 1]

                    # 取第一個預測值
                    if pred.dim() == 2 and pred.size(1) > 1:
                        next_pred = pred[:, 0:1]  # [batch, 1]
                    else:
                        next_pred = pred.unsqueeze(-1) if pred.dim() == 1 else pred

                    rolling_preds.append(next_pred.squeeze(-1))  # [batch]

                    # ========== 更新輸入序列 ==========
                    next_step = torch.zeros(batch_size, 1, n_features, device=self.device)
                    next_step[:, 0, 0] = next_pred.squeeze()  # log_close

                    # 其他特徵用最後一個時間步的值
                    next_step[:, 0, 1:] = current_input[:, -1, 1:]

                    # 滾動更新
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        next_step
                    ], dim=1)

                rolling_preds = torch.stack(rolling_preds, dim=1)  # [batch, pred_len]

                all_preds_rolling.append(rolling_preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                all_ticker_ids.append(tid_batch.numpy())

        all_preds_scaled = np.concatenate(all_preds_rolling)
        all_targets_scaled = np.concatenate(all_targets)
        all_ticker_ids = np.concatenate(all_ticker_ids)

        # ========== 反標準化 ==========
        preds_price, targets_price, ticker_names = self._inverse_transform(
            all_preds_scaled, all_targets_scaled, all_ticker_ids,
            feature_cols, scalers, id2ticker, pred_len
        )

        # ========== 計算指標 ==========
        # 計算整體 MSE (scaled)
        criterion = nn.MSELoss()
        avg_test_loss_scaled = criterion(
            torch.tensor(all_preds_scaled), 
            torch.tensor(all_targets_scaled)
        ).item()

        results = self._calculate_metrics(
            preds_price, targets_price, ticker_names, 
            avg_test_loss_scaled, model_name, pred_len
        )

        # ========== 繪圖 ==========
        self._plot_predictions(
            preds_price, targets_price, ticker_names, all_ticker_ids,
            id2ticker, all_dates, has_dates, model_name, pred_len, mode="autoregressive"
        )

        return results

    def _inverse_transform(self, preds_scaled, targets_scaled, ticker_ids, 
                          feature_cols, scalers, id2ticker, pred_len):
        """反標準化 + exp"""
        preds_price_list = []
        targets_price_list = []
        ticker_names = []

        D = len(feature_cols)

        for s_pred, s_true, tid in zip(preds_scaled, targets_scaled, ticker_ids):
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

        return preds_price, targets_price, ticker_names

    def _calculate_metrics(self, preds_price, targets_price, ticker_names, 
                          avg_test_loss_scaled, model_name, pred_len):
        """計算所有評估指標"""
        # 整體指標
        mse_price = np.mean((preds_price - targets_price) ** 2)
        rmse_price = math.sqrt(mse_price)
        mae_price = np.mean(np.abs(preds_price - targets_price))
        mape_price = np.mean(np.abs((targets_price - preds_price) / targets_price)) * 100

        # R² (加權平均)
        r2_per_stock = []
        stock_counts = []
        
        for ticker in sorted(set(ticker_names)):
            ticker_mask = np.array(ticker_names) == ticker
            ticker_preds = preds_price[ticker_mask]
            ticker_targets = targets_price[ticker_mask]
            
            ticker_r2 = self.calculate_r2_score(ticker_targets, ticker_preds)
            r2_per_stock.append(ticker_r2)
            stock_counts.append(len(ticker_preds))
        
        total_samples = sum(stock_counts)
        r2_weighted = sum(r2 * count for r2, count in zip(r2_per_stock, stock_counts)) / total_samples
        r2_mixed = self.calculate_r2_score(targets_price, preds_price)

        # 顯示整體結果
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
        print(f"  Mixed Stocks R²:            {r2_mixed:.4f}  ⚠️  Reference only")
        
        if pred_len > 1:
            print(f"\n{'─'*60}")
            print(f"Per-Day Average Metrics:")
            for day in range(pred_len):
                if preds_price.ndim == 2:
                    day_mape = np.mean(np.abs((targets_price[:, day] - preds_price[:, day]) / targets_price[:, day])) * 100
                    day_r2 = self.calculate_r2_score(targets_price[:, day], preds_price[:, day])
                    print(f"  Day {day+1}: MAPE={day_mape:.2f}%, R²={day_r2:.4f}")
        
        print(f"{'='*60}\n")

        # 分股票結果
        print(f"{'='*60}")
        print(f"Per-Stock Test Results ({model_name})")
        print(f"{'='*60}\n")

        results = {
            "overall": {
                "mse": float(mse_price),
                "rmse": float(rmse_price),
                "mae": float(mae_price),
                "mape": float(mape_price),
                "r2_weighted": float(r2_weighted),
                "r2_mixed": float(r2_mixed),
            },
            "per_stock": {},
        }

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
            
            ticker_results = {
                "mse": float(ticker_mse),
                "rmse": float(ticker_rmse),
                "mae": float(ticker_mae),
                "mape": float(ticker_mape),
                "r2": float(ticker_r2),
            }
            
            if pred_len > 1 and ticker_preds.ndim == 2:
                print(f"  Per-Day Metrics:")
                ticker_results["per_day_mape"] = {}
                ticker_results["per_day_r2"] = {}
                for day in range(pred_len):
                    day_mape = np.mean(np.abs((ticker_targets[:, day] - ticker_preds[:, day]) / ticker_targets[:, day])) * 100
                    day_r2 = self.calculate_r2_score(ticker_targets[:, day], ticker_preds[:, day])
                    ticker_results["per_day_mape"][f"day_{day+1}"] = float(day_mape)
                    ticker_results["per_day_r2"][f"day_{day+1}"] = float(day_r2)
                    print(f"    Day {day+1}: MAPE={day_mape:.2f}%, R²={day_r2:.4f}")
            
            print()
            results["per_stock"][ticker] = ticker_results

        return results

    def _plot_predictions(self, preds_price, targets_price, ticker_names, ticker_ids,
                         id2ticker, all_dates, has_dates, model_name, pred_len, mode="static"):
        """繪製預測圖"""
        print(f"{'='*60}")
        print(f"Generating Prediction Plots ({model_name} - {mode.upper()})")
        print(f"{'='*60}\n")

        for ticker in sorted(set(ticker_names)):
            ticker_id = None
            for tid, tname in id2ticker.items():
                if tname == ticker:
                    ticker_id = tid
                    break

            if ticker_id is not None:
                mask = ticker_ids == ticker_id
                idx = np.where(mask)[0]

                if len(idx) > 0:
                    n_plot = min(100, len(idx))
                    sel_idx = idx[:n_plot]

                    # 準備 x 軸
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
                        self._plot_single_step(
                            x_axis, targets_price[sel_idx], preds_price[sel_idx],
                            ticker, ticker_names, model_name, use_date_format, x_label, mode
                        )
                    else:
                        self._plot_multi_step(
                            x_axis, targets_price[sel_idx], preds_price[sel_idx],
                            ticker, ticker_names, model_name, pred_len, 
                            use_date_format, x_label, mode
                        )

        print(f"\n{'='*60}\n")

    def _plot_single_step(self, x_axis, targets, preds, ticker, ticker_names, 
                         model_name, use_date_format, x_label, mode):
        """單步預測圖"""
        fig = plt.figure(figsize=(16, 6))
        
        ax1 = plt.subplot(1, 2, 1)
        plt.plot(x_axis, targets, label=f"{ticker} True", linewidth=2.5, alpha=0.9, color='#2E86AB', marker='o', markersize=4)
        plt.plot(x_axis, preds, label=f"{ticker} Pred ({model_name})", linewidth=2, alpha=0.8, color='#A23B72', marker='x', markersize=4)
        
        if use_date_format:
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(MonthLocator(interval=2))
            plt.xticks(rotation=45, ha='right')
        
        ticker_mask = np.array(ticker_names) == ticker
        ticker_r2 = self.calculate_r2_score(targets, preds)
        
        plt.legend(loc='best', fontsize=10, title=f"R² = {ticker_r2:.4f}")
        plt.title(f"{ticker} - Prediction ({model_name}, {mode.upper()})", fontsize=12, fontweight='bold')
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel("Price", fontsize=10)
        plt.grid(True, alpha=0.3)

        ax2 = plt.subplot(1, 2, 2)
        errors = preds - targets
        plt.plot(x_axis, errors, linewidth=2, color='#F18F01', alpha=0.8)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        
        if use_date_format:
            ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(MonthLocator(interval=2))
            plt.xticks(rotation=45, ha='right')
        
        plt.title(f"{ticker} - Errors ({model_name})", fontsize=12, fontweight='bold')
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel("Error", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        plot_path = f"output/picture/{model_name.lower()}_{ticker.replace('.', '_')}_{mode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[{ticker}] ✓ Plot saved to {plot_path}")
        plt.close()

    def _plot_multi_step(self, x_axis, targets, preds, ticker, ticker_names,
                        model_name, pred_len, use_date_format, x_label, mode):
        """多步預測圖"""
        fig = plt.figure(figsize=(18, 12))
        
        # 子圖1: 預測 vs 真實
        ax1 = plt.subplot(2, 1, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, pred_len))
        
        plt.plot(x_axis, targets[:, 0], label=f"{ticker} True (Day 1)", 
                linewidth=3, alpha=0.9, color='black', marker='o', markersize=4)
        
        for day in range(pred_len):
            plt.plot(x_axis, preds[:, day], label=f"Pred Day {day+1} ({mode.upper()})",
                    linewidth=2, alpha=0.7, color=colors[day], marker='x', markersize=3)
        
        if use_date_format:
            ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(MonthLocator(interval=2))
            plt.xticks(rotation=45, ha='right')
        
        ticker_mask = np.array(ticker_names) == ticker
        ticker_r2 = self.calculate_r2_score(targets, preds)
        
        plt.legend(loc='best', fontsize=9, ncol=2, title=f"R² = {ticker_r2:.4f}")
        plt.title(f"{ticker} - Multi-Step Predictions ({model_name}, {mode.upper()})", 
                 fontsize=13, fontweight='bold')
        plt.xlabel(x_label, fontsize=11)
        plt.ylabel("Price", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 子圖2: 每日MAPE和R²
        ax2 = plt.subplot(2, 1, 2)
        day_mapes = []
        day_r2s = []
        for day in range(pred_len):
            day_mape = np.mean(np.abs((targets[:, day] - preds[:, day]) / targets[:, day])) * 100
            day_r2 = self.calculate_r2_score(targets[:, day], preds[:, day])
            day_mapes.append(day_mape)
            day_r2s.append(day_r2)
        
        x_pos = np.arange(1, pred_len + 1)
        width = 0.35
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x_pos - width/2, day_mapes, width, color=colors, alpha=0.8, label='MAPE', edgecolor='black')
        bars2 = ax2_twin.bar(x_pos + width/2, day_r2s, width, color='gray', alpha=0.6, label='R²', edgecolor='black')
        
        for bar, val in zip(bars1, day_mapes):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, day_r2s):
            ax2_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel("Prediction Day", fontsize=11)
        ax2.set_ylabel("MAPE (%)", fontsize=11, color='blue')
        ax2_twin.set_ylabel("R²", fontsize=11, color='red')
        ax2.set_title(f"{ticker} - Performance by Day ({mode.upper()})", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(x_pos)
        
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        plot_path = f"output/picture/{model_name.lower()}_{ticker.replace('.', '_')}_{mode}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[{ticker}] ✓ Plot saved to {plot_path}")
        plt.close()