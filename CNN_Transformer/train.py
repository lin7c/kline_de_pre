import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
from Dmodel import GafCnnTransformer
from sklearn.metrics import r2_score


def train_model(MODEL_PATH, RESUME, X_FILE, Y_FILE, output_dim=9, epochs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # 1. 加载数据
    if not os.path.exists(X_FILE) or not os.path.exists(Y_FILE):
        raise FileNotFoundError(f"找不到数据文件: {X_FILE} 或 {Y_FILE}")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    min_samples = min(len(X), len(y))
    X = X[:min_samples]
    y = y[:min_samples]

    print(f">>> 数据加载完成 | 样本数: {min_samples}")
    print(f">>> X shape: {X.shape} | Y shape: {y.shape}")
    print(
        f">>> Y 统计信息 (应接近 mean≈0, std≈1): mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")

    # 直接使用 makedata.py 生成的标准化后的 Y
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(X_tensor, y_tensor)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # 2. 模型、优化器、损失
    model = GafCnnTransformer(output_dim=output_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.HuberLoss(delta=1.0)  # 在标准化空间建议使用 1.0 左右

    best_val_loss = float('inf')
    start_epoch = 0

    # 3. Resume 逻辑（简化）
    if RESUME and os.path.exists(MODEL_PATH):
        print(f">>> 从 {MODEL_PATH} 恢复训练...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f">>> 已恢复，最佳 val_loss: {best_val_loss:.5f}")
        else:
            model.load_state_dict(checkpoint)

    # 4. 训练循环
    early_stop_patience = 100
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 在标准化后的 DCT 空间计算 R²（干净且合理）
        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        current_r2 = r2_score(y_true, y_pred)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1:03d}/{epochs}] | LR: {optimizer.param_groups[0]['lr']:.7f} | "
              f"Loss(T/V): {avg_train_loss:.5f}/{avg_val_loss:.5f} | R²: {current_r2:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint_data, MODEL_PATH)
            print(f">>> 保存最佳模型 (val_loss: {best_val_loss:.5f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"!!! 早停触发（连续 {early_stop_patience} 个 epoch 无改善）")
                break
    for i in range(y.shape[1]):
        col_mean = np.mean(y[:, i])
        col_std = np.std(y[:, i])
        col_max = np.max(y[:, i])
        col_min = np.min(y[:, i])
        print(f"维度 {i:02d} | 均值: {col_mean:8.4f} | 标准差: {col_std:8.4f} | 范围: [{col_min:8.2f}, {col_max:8.2f}]")
    print(">>> 训练完成！")
    return model


def run():
    CONFIG = {
        "MODEL_PATH": "transformer_dct_v1.pth",
        "RESUME": True,  # 换数据集或想重新训练时设为 False
        "X_FILE": "../RL/gaf_v1.npy",
        "Y_FILE": "y_transformer_v1.npy",
        "output_dim": 9,
        "epochs": 10000
    }
    train_model(**CONFIG)


if __name__ == "__main__":
    run()