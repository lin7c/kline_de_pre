import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct
import random


def decode_and_plot(X_path, Y_path, n_samples=3, n_components=3):
    # 1. 加载数据
    # X: (N, 60, 12) 原始价格数据
    # Y: (N, 9) 包含 1m, 5m, 15m 的 DCT 系数 (每个 3 个)
    X = np.load(X_path)
    Y = np.load(Y_path)

    sample_indices = random.sample(range(len(X)), n_samples)

    # 配置信息：必须与 generate_dct_trend_labels 中的 step/gap 对应
    configs = [
        {"name": "1m", "x_col": 3, "gap": 60, "step": 1},
        {"name": "5m", "x_col": 7, "gap": 300, "step": 5},
        {"name": "15m", "x_col": 11, "gap": 900, "step": 15}
    ]

    fig, axes = plt.subplots(n_samples, 3, figsize=(20, 5 * n_samples))

    for row, idx in enumerate(sample_indices):
        for col, cfg in enumerate(configs):
            ax = axes[row, col]

            gap = cfg["gap"]
            x_col = cfg["x_col"]
            name = cfg["name"]

            # --- A. 提取真实未来走势 ---
            true_future = []
            for t in range(1, gap + 1):
                if idx + t < len(X):
                    true_future.append(X[idx + t, -1, x_col])
            true_future = np.array(true_future)

            # --- B. 核心还原算法 (Symmetric Reconstruction) ---
            start = col * n_components
            coeffs = Y[idx, start: start + n_components]

            # 关键：生成端压缩到了 60 点，所以还原端必须先固定还原为 60 点
            fixed_recon_len = 60
            full_coeffs = np.zeros(fixed_recon_len)
            full_coeffs[:n_components] = coeffs

            # 执行正交 IDCT
            restored_norm = idct(full_coeffs, type=2, norm='ortho')

            # 强制起点对齐：计算 IDCT 序列第一个值与 0 的偏差并平移
            # 因为我们在编码时用的是 (compressed_series - base_price)
            # 理想情况下 restored_norm[0] 应该接近 0
            offset = restored_norm[0]
            restored_norm = restored_norm - offset

            # 乘回局部标准差 (Local Scaling)
            local_std = np.std(X[idx, :, x_col]) + 1e-9
            restored_diff_60 = restored_norm * local_std

            # 叠加基准价格
            base_price = X[idx, -1, x_col]
            restored_trend_60 = restored_diff_60 + base_price

            # --- C. 时间轴映射 ---
            # 历史轴：-59 到 0
            past_x = np.arange(-59, 1)
            # 未来轴：将 60 个预测点均匀分布到真实的 gap 长度上
            future_x_true = np.arange(1, len(true_future) + 1)
            future_x_pred = np.linspace(1, gap, fixed_recon_len)

            # --- D. 绘图 ---
            # 绘制历史
            ax.plot(past_x, X[idx, :, x_col], color='black', alpha=0.3, label='History')
            # 绘制真实未来
            if len(true_future) > 0:
                ax.plot(future_x_true, true_future, color='green', alpha=0.4, label='True Future')
            # 绘制 DCT 趋势还原
            ax.plot(future_x_pred, restored_trend_60, color='red', linestyle='--', linewidth=2,
                    label=f'DCT Trend (k={n_components})')

            # 标注当前时刻
            ax.axvline(0, color='blue', linestyle=':', alpha=0.5)
            ax.scatter(0, base_price, color='red', s=30, zorder=5)  # 起点标记

            ax.set_title(f"Sample {idx} | {name} | std={local_std:.4f}")
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize='small')

    plt.suptitle("Symmetric DCT Reconstruction: Fixed 60-pt IDCT with Start-Point Alignment", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # 注意：请确保 Y_path 指向的是由你刚才提供的 generate_dct_trend_labels 生成的文件
    decode_and_plot(
        X_path="../CNN/input_x_v1.npy",
        Y_path="y_transformer_v1.npy",
        n_samples=3,
        n_components=3
    )