import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os

# 配置参数
SYMBOL = "ETHUSDm"
TIMEFRAMES = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15
}
WINDOW_SIZE = 60
# 设置一个足够大的 FETCH_COUNT，或者根据 MT5 允许的最大值获取
FETCH_COUNT = 5000


def get_data():
    if not mt5.initialize():
        print("MT5 初始化失败")
        return None

    data_frames = {}
    for name, tf in TIMEFRAMES.items():
        # 这里尝试获取尽可能多的数据
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, FETCH_COUNT)
        if rates is None:
            print(f"无法获取 {name} 数据")
            continue
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close']]
        df.columns = ['time'] + [f"{name}_{c}" for c in ['open', 'high', 'low', 'close']]
        data_frames[name] = df

    mt5.shutdown()

    # 以 1m 为基准合并
    base_df = data_frames['1m']
    combined_df = pd.merge_asof(base_df, data_frames['5m'], on='time', direction='backward')
    combined_df = pd.merge_asof(combined_df, data_frames['15m'], on='time', direction='backward')

    # 关键：dropna 后剩下的就是所有可以用于生成窗口的有效连续数据
    combined_df.dropna(inplace=True)
    return combined_df


def create_max_sliding_windows(df, window_size):
    # 转换为 numpy
    data_values = df.drop(columns=['time']).values
    total_len = len(data_values)

    if total_len < window_size:
        raise ValueError(f"数据量不足，总长度 {total_len} 小于窗口大小 {window_size}")

    # --- 核心修改：动态计算可生成的最大窗口数 ---
    # 可生成的数量 = 总长度 - 窗口大小 + 1
    num_windows = total_len - window_size + 1

    # 使用 np.lib.stride_tricks 实现超快速滑动窗口生成（不再需要循环）
    # 形状：(num_windows, window_size, features)
    shape = (num_windows, window_size, data_values.shape[1])
    strides = (data_values.strides[0], data_values.strides[0], data_values.strides[1])
    windows = np.lib.stride_tricks.as_strided(data_values, shape=shape, strides=strides)

    return windows, num_windows


if __name__ == "__main__":
    print(f"正在从 MT5 获取 {SYMBOL} 数据...")
    full_df = get_data()

    if full_df is not None:
        # 记录原始数据总行数
        total_data_points = len(full_df)
        print(f"原始对齐数据共: {total_data_points} 条")

        # 尽可能多地生成窗口
        windows_data, actual_num = create_max_sliding_windows(full_df, WINDOW_SIZE)

        # 保存数据
        output_file = "ethusdm_multi_tf_windows.npy"
        np.save(output_file, windows_data)

        # 保存对齐后的全量 CSV (后续提取时间戳需要用到)
        full_df.to_csv("ethusdm_aligned_data.csv", index=False)

        print(f"成功！")
        print(f"可生成的最大窗口数量: {actual_num}")
        print(f"窗口形状: {windows_data.shape}")