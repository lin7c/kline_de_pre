import numpy as np
import matplotlib.pyplot as plt


def visualize_gaf(file_path="../RL/gaf_v1.npy", sample_idx=None):
    # 1. 加载数据
    # 假设形状为 (N, 12, 60, 60) 或 (N, 60, 60, 12)
    data = np.load(file_path)
    print(f"数据形状: {data.shape}")

    # 2. 调整维度到 (N, 60, 60, 12) 方便绘图
    if data.shape[1] == 12:
        data = data.transpose(0, 2, 3, 1)

    # 3. 随机选择一个样本
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(data))

    sample = data[sample_idx]

    # 4. 定义通道名称
    channels = [
        "1m_Open", "1m_High", "1m_Low", "1m_Close",
        "5m_Open", "5m_High", "5m_Low", "5m_Close",
        "15m_Open", "15m_High", "15m_Low", "15m_Close"
    ]

    # 5. 绘图：展示 3行4列 (12个通道)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"Sample Index: {sample_idx} - GAF 12 Channels Visualization", fontsize=20)

    for i in range(12):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        # 使用 'rainbow' 或 'viridis' 颜色映射，观察纹理更清晰
        im = ax.imshow(sample[:, :, i], cmap='rainbow', origin='lower')
        ax.set_title(channels[i])
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == "__main__":
    # 替换为你实际的 gaf 文件路径
    visualize_gaf("../RL/gaf_v1.npy")