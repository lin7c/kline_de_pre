import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# --- 参数设置 ---
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDm"
WINDOW_SIZE = 60
# 尝试获取的最大 K 线数量（MT5 实际上受限于工具 -> 选项 -> 图表中的“最大柱数”）
MAX_FETCH = 1000000

TIMEFRAMES = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15
}


def get_data():
    if not mt5.initialize():
        print("❌ MT5 初始化失败")
        return None

    print(f"📡 正在获取 {SYMBOL} 的全量多周期数据...")
    data_frames = {}

    for name, tf in TIMEFRAMES.items():
        # 获取该周期下的最大可用数据
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, MAX_FETCH)

        if rates is None or len(rates) == 0:
            print(f"⚠️ 无法获取 {name} 周期数据")
            continue

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # 提取并重命名列
        df = df[['time', 'open', 'high', 'low', 'close']]
        df.columns = ['time'] + [f"{name}_{c}" for c in ['open', 'high', 'low', 'close']]

        # 确保按时间升序排列
        df = df.sort_values('time').reset_index(drop=True)
        data_frames[name] = df
        print(f"✅ {name} 获取成功: {len(df)} 根")

    mt5.shutdown()

    # 核心：必须有 1m 数据作为对齐基准
    if "1m" not in data_frames:
        print("❌ 错误：未获取到 1m 基准数据")
        return None

    # --- 跨周期精准对齐 ---
    print("🔄 正在进行跨周期时间对齐...")
    combined_df = data_frames['1m']

    # 将 5m 和 15m 对齐到 1m 的时间轴上
    # direction='backward' 确保了在 1m 的任何时刻，只能看到已经结束的大周期 K 线（防止未来函数）
    if "5m" in data_frames:
        combined_df = pd.merge_asof(combined_df, data_frames['5m'], on='time', direction='backward')
    if "15m" in data_frames:
        combined_df = pd.merge_asof(combined_df, data_frames['15m'], on='time', direction='backward')

    # 删除因为大周期数据尚未开始而产生的 NaN 行
    initial_count = len(combined_df)
    combined_df.dropna(inplace=True)
    final_count = len(combined_df)

    print(f"📊 对齐完成。原始 1m: {initial_count} 行 -> 有效对齐: {final_count} 行")
    return combined_df


def create_max_sliding_windows(df, window_size):
    """ 使用 Stride Tricks 创建滑动窗口 """
    # 移除时间列，只保留数值
    data_values = df.drop(columns=['time']).values
    total_len = len(data_values)

    if total_len < window_size:
        raise ValueError("有效数据量小于窗口大小")

    num_windows = total_len - window_size + 1

    # 计算步长形状
    shape = (num_windows, window_size, data_values.shape[1])
    strides = (data_values.strides[0], data_values.strides[0], data_values.strides[1])

    windows = np.lib.stride_tricks.as_strided(data_values, shape=shape, strides=strides)
    return windows, num_windows


def run(output_file="org_v1.npy", csv_file="org_v1.csv"):
    full_df = get_data()

    if full_df is not None:
        try:
            windows_data, actual_num = create_max_sliding_windows(full_df, WINDOW_SIZE)

            # 保存结果
            np.save(output_file, windows_data)
            full_df.to_csv(csv_file, index=False)

            print("-" * 30)
            print(f"✅ 任务成功完成！")
            print(f"📂 导出文件: {output_file} & {csv_file}")
            print(f"📐 最终形状: {windows_data.shape} (样本数, 窗口长度, 特征数)")
            print(f"🕒 时间范围: {full_df['time'].iloc[0]} 至 {full_df['time'].iloc[-1]}")
            print("-" * 30)
        except Exception as e:
            print(f"❌ 处理窗口时出错: {e}")
            sys.exit(1)
    else:
        print(f"❌ {SYMBOL} 数据获取失败。")
        sys.exit(1)


if __name__ == "__main__":
    run()