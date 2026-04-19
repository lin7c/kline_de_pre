import numpy as np
from pyts.image import GramianAngularField
import os


def convert_raw_to_gaf(input_path, output_path):
    # 1. 加载原始 3D 数据 (N, 60, 12)
    if not os.path.exists(input_path):
        print(f"错误：找不到输入文件 {input_path}")
        return

    X_raw = np.load(input_path)
    num_samples, window_size, num_channels = X_raw.shape
    print(f"检测到原始数据形状: {X_raw.shape}")

    # 2. 初始化 GAF 转换器
    # method='summation' 即 GASF，适合捕捉趋势和形态
    gaf = GramianAngularField(method='summation', sample_range=(-1, 1))

    # 3. 创建输出容器 (N, H, W, C)
    # 使用 float32 足够满足深度学习需求且节省内存
    X_gaf = np.empty((num_samples, num_channels, window_size, window_size), dtype=np.float32)

    print("开始执行 GAF 转换...")

    for i in range(num_samples):
        # 遍历 12 个通道 (1m_OHLC, 5m_OHLC, 15m_OHLC)
        for ch in range(num_channels):
            # 提取序列 (60,)
            series = X_raw[i, :, ch].reshape(1, -1)

            # --- 关键：局部窗口归一化 ---
            # GAF 强制要求数据在 [-1, 1] 之间。
            # 我们对每个样本的每个通道独立归一化，保留其几何轮廓
            s_min, s_max = series.min(), series.max()
            if s_max - s_min < 1e-9:
                series_norm = np.zeros_like(series)
            else:
                series_norm = (series - s_min) / (s_max - s_min) * 2 - 1

            # 转换并存入 4D 张量的对应通道
            # gaf.fit_transform 返回 (1, 60, 60)，取 [0] 得到 (60, 60)
            X_gaf[i, ch, :, :] = gaf.fit_transform(series_norm)[0]

        if (i + 1) % 500 == 0:
            print(f"已完成样本数: {i + 1}/{num_samples}")

    # 4. 保存结果
    np.save(output_path, X_gaf)
    print("-" * 30)
    print(f"转换成功！")
    print(f"输出文件: {output_path}")
    print(f"最终张量形状: {X_gaf.shape} (样本, 高, 宽, 通道)")


def run(
        input_file = "input_x_v1.npy",
        output_file = "../RL/gaf_v1.npy",
    ):
    convert_raw_to_gaf(input_file, output_file)
if __name__ == "__main__":
    run()