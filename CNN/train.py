import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
from TPmodel import GafRegressionCNN
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_model(X_path, y_path, model_save_path="best_gaf_cnn_model.pth",
                batch_size=32, epochs=10000, resume=True):
    # --- 加载数据（不做任何全局标准化）---
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"找不到数据文件: {X_path} 或 {y_path}")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f">>> 数据加载完成 | X shape: {X.shape} | y shape: {y.shape}")
    print(
        f">>> y 统计 (相对变化率): mean={np.mean(y):.5f}, std={np.std(y):.5f}, min={np.min(y):.5f}, max={np.max(y):.5f}")

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_db, val_db = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)

    # --- 初始化模型 ---
    model = GafRegressionCNN().to(device)

    # --- Resume 逻辑 ---
    initial_lr = 0.001
    if resume and os.path.exists(model_save_path):
        print(f">>> 加载预训练模型 {model_save_path} 进行微调...")
        model.load_state_dict(torch.load(model_save_path, map_location=device,weights_only=False))
        initial_lr = 0.0005
        print(f">>> 学习率调整为: {initial_lr}")
    else:
        print(">>> 从零开始训练。")

    criterion = nn.HuberLoss(delta=0.01)  # 相对变化率建议 delta 小一点
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    best_val_loss = float('inf')
    early_stop_patience = 100
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # 验证
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                all_outputs.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.concatenate(all_outputs, axis=0)
        current_r2 = r2_score(y_true, y_pred)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | R²: {current_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f">>> Best Model Saved! Val Loss: {best_val_loss:.6f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"!!! 早停触发（连续 {early_stop_patience} 个 epoch 无改善）")
                break

    return model


def run():
    train_model(
        X_path="../RL/gaf_v1.npy",  # 注意：这里应该用你生成好的 GAF 数据
        y_path="y_cnn_v1.npy",
        model_save_path="cnn_model_v1.pth",
        batch_size=32,
        epochs=10000,
        resume=True  # 换数据时可改为 False
    )


if __name__ == "__main__":
    run()