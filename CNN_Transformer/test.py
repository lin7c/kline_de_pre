import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.fftpack import idct
from Dmodel import GafCnnTransformer


def run_inference_and_visualize(
        gaf_path="../RL/gaf_v1.npy",
        raw_x_path="../org_v1.npy",
        model_path="transformer_dct_v1.pth",
        n_samples=3,
        n_components=3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载模型 ---
    model = GafCnnTransformer(output_dim=9).to(device)
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint,
                                                              dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f">>> 模型加载成功: {model_path}")

    # --- 2. 加载数据 ---
    X_gaf = np.load(gaf_path)
    X_raw = np.load(raw_x_path)[:len(X_gaf)]  # 确保样本数对齐

    # --- 3. 随机取样 ---
    total_samples = len(X_gaf)
    sample_indices = random.sample(range(total_samples), n_samples)

    # 提取选中的样本并转为 Tensor
    selected_gaf = torch.from_numpy(X_gaf[sample_indices]).float().to(device)
    print(selected_gaf[0, 0, 0, :5])
    # --- 4. 执行推理 ---
    with torch.no_grad():
        # y_pred 形状: (n_samples, 9) -> 标准化后的 DCT 系数
        y_pred = model(selected_gaf).cpu().numpy()

    # --- 5. 配置与绘图准备 ---
    configs = [
        {"name": "1m", "x_col": 3, "gap": 60},
        {"name": "5m", "x_col": 7, "gap": 300},
        {"name": "15m", "x_col": 11, "gap": 900}
    ]

    fig, axes = plt.subplots(n_samples, 3, figsize=(20, 5 * n_samples))
    if n_samples == 1: axes = np.expand_dims(axes, axis=0)  # 确保 axes 是 2D 的

    # --- 6. 循环还原并绘图 ---
    for row, idx in enumerate(sample_indices):
        for col, cfg in enumerate(configs):
            ax = axes[row, col]
            x_col = cfg["x_col"]
            gap = cfg["gap"]
            name = cfg["name"]

            # A. 提取预测系数并还原局部波动率 (local_std)
            start_idx = col * n_components
            pred_coeffs_norm = y_pred[row, start_idx: start_idx + n_components]

            # 计算原始数据的 local_std (基于 X_raw)
            local_std = np.std(X_raw[idx, :, x_col]) + 1e-9
            pred_coeffs = pred_coeffs_norm * local_std

            # B. IDCT 还原 (固定 60 点对称还原)
            fixed_recon_len = 60
            full_coeffs = np.zeros(fixed_recon_len)
            full_coeffs[:n_components] = pred_coeffs

            restored_norm = idct(full_coeffs, type=2, norm='ortho')

            # 强制起点对齐 (Zero-offset alignment)
            restored_norm = restored_norm - restored_norm[0]

            # 叠加当前时刻基准价格
            base_price = X_raw[idx, -1, x_col]
            restored_trend = restored_norm + base_price

            # C. 提取真实未来走势 (用于对比)
            true_future = []
            for t in range(1, gap + 1):
                if idx + t < total_samples:
                    true_future.append(X_raw[idx + t, -1, x_col])
            true_future = np.array(true_future)

            # D. 坐标轴设置
            past_x = np.arange(-59, 1)
            future_x_true = np.arange(1, len(true_future) + 1)
            future_x_pred = np.linspace(1, gap, fixed_recon_len)

            # E. 绘图
            ax.plot(past_x, X_raw[idx, :, x_col], color='black', alpha=0.3, label='History')
            if len(true_future) > 0:
                ax.plot(future_x_true, true_future, color='green', alpha=0.4, label='True Future')
            ax.plot(future_x_pred, restored_trend, color='red', linestyle='--', linewidth=2,
                    label=f'Pred DCT (k={n_components})')

            ax.axvline(0, color='blue', linestyle=':', alpha=0.5)
            ax.scatter(0, base_price, color='red', s=30)

            ax.set_title(f"Idx {idx} | {name} | std={local_std:.4f}")
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0: ax.legend()

    plt.suptitle("Model Inference & IDCT Reconstruction (Random Samples)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_inference_and_visualize(
        gaf_path="../RL/gaf_v1.npy",
        raw_x_path="../org_v1.npy",
        model_path="transformer_dct_v1.pth",
        n_samples=3,
        n_components=3
    )