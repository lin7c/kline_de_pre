import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from TPmodel import GafRegressionCNN


def test_peak_prediction(gaf_path="../RL/gaf_v1.npy",
                         X_raw_path="../CNN/input_x_v1.npy",
                         Y_path="y_cnn_v1.npy",
                         model_path="cnn_model_v1.pth",
                         n_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    X_gaf_all = np.load(gaf_path)  # (N, 12, 60, 60)
    X_raw_all = np.load(X_raw_path)  # (N, 60, 12)
    Y_true_norm = np.load(Y_path)  # 已经是局部标准化后的标签 (N, 6)

    # 2. 加载模型及潜在的全局标准化参数
    model = GafRegressionCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 如果你在训练中加入了全局标准化，需要从 checkpoint 提取
    y_mean = 0
    y_std = 1
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # 自动适配全局标准化参数
            y_mean = checkpoint.get('y_mean', 0)
            y_std = checkpoint.get('y_std', 1)
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f">>> 模型加载成功。全局标准化适配: mean={np.mean(y_mean):.4f}, std={np.mean(y_std):.4f}")

    # 3. 随机抽样
    indices = random.sample(range(len(X_gaf_all)), n_samples)

    # 窗口配置 (y_idx 对应 y_cnn_v1.npy 中的 6 个维度)
    configs = [
        {"name": "1m", "x_col": 3, "y_idx": (0, 1), "gap": 60},
        {"name": "5m", "x_col": 7, "y_idx": (2, 3), "gap": 300},
        {"name": "15m", "x_col": 11, "y_idx": (4, 5), "gap": 900}
    ]

    fig, axes = plt.subplots(n_samples, 3, figsize=(22, 5 * n_samples))

    for row, idx in enumerate(indices):
        input_tensor = torch.from_numpy(X_gaf_all[idx]).unsqueeze(0).to(device).float()

        with torch.no_grad():
            # 得到模型输出 (可能是全局标准化后的)
            preds_raw = model(input_tensor).cpu().numpy()[0]
            # 1. 逆全局标准化 -> 得到局部标准化空间的值
            preds_norm = preds_raw * y_std + y_mean

        current_x_raw = X_raw_all[idx]

        # --- 核心还原逻辑：获取局部标准化因子 ---
        # 必须与 makedata.py 中的逻辑严格一致：使用 1m Close 的历史 std
        local_std = np.std(current_x_raw[:, 3]) + 1e-9
        curr_p = current_x_raw[-1, 3]  # 以当前 1m 收盘价为基准

        for col, cfg in enumerate(configs):
            ax = axes[row, col]

            # 2. 逆局部标准化 -> 还原价格
            # 价格 = 当前价 + (标准化值 * 历史波动率)
            p_h = curr_p + (preds_norm[cfg['y_idx'][0]] * local_std)
            p_l = curr_p + (preds_norm[cfg['y_idx'][1]] * local_std)

            t_h = curr_p + (Y_true_norm[idx, cfg['y_idx'][0]] * local_std)
            t_l = curr_p + (Y_true_norm[idx, cfg['y_idx'][1]] * local_std)

            # 获取未来走势
            future_actual = []
            for t in range(1, cfg['gap'] + 1):
                if idx + t < len(X_raw_all):
                    future_actual.append(X_raw_all[idx + t, -1, cfg['x_col']])

            # 绘图
            ax.plot(np.arange(-59, 1), current_x_raw[:, cfg['x_col']], color='black', alpha=0.7, label='History')
            if len(future_actual) > 0:
                ax.plot(np.linspace(1, 60, len(future_actual)), future_actual, color='blue', alpha=0.4, label='Future')

            # 预测线 (虚线)
            ax.axhline(p_h, color='red', linestyle='--', linewidth=1.2,
                       label=f'Pred H (Std:{preds_norm[cfg["y_idx"][0]]:.2f})')
            ax.axhline(p_l, color='green', linestyle='--', linewidth=1.2,
                       label=f'Pred L (Std:{preds_norm[cfg["y_idx"][1]]:.2f})')

            # 真实区间 (浅色背景带)
            ax.axhline(t_h, color='red', alpha=0.15, linewidth=3)
            ax.axhline(t_l, color='green', alpha=0.15, linewidth=3)

            ax.set_title(f"Idx {idx} | {cfg['name']} | LocalStd: {local_std:.2f}")
            if row == 0 and col == 0: ax.legend(loc='upper left', fontsize='xx-small')
            ax.grid(True, alpha=0.1)

    plt.suptitle("Peak Prediction - Local Normalization Inverse Mapping", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    test_peak_prediction()