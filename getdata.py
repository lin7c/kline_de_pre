import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
import sys  # 引入 sys 获取外部参数

# --- 修改：增加参数解析 ---
# 默认值，如果没有传参数则使用 ETHUSDm
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDm"

TIMEFRAMES = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15
}
WINDOW_SIZE = 60
FETCH_COUNT = 10000


def get_data():
    if not mt5.initialize():
        print(f"MT5 初始化失败")
        return None

    data_frames = {}
    for name, tf in TIMEFRAMES.items():
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, FETCH_COUNT)
        if rates is None:
            print(f"无法获取 {SYMBOL} 的 {name} 数据")
            continue
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close']]
        df.columns = ['time'] + [f"{name}_{c}" for c in ['open', 'high', 'low', 'close']]
        data_frames[name] = df

    mt5.shutdown()

    if "1m" not in data_frames: return None

    base_df = data_frames['1m']
    combined_df = pd.merge_asof(base_df, data_frames['5m'], on='time', direction='backward')
    combined_df = pd.merge_asof(combined_df, data_frames['15m'], on='time', direction='backward')
    combined_df.dropna(inplace=True)
    return combined_df


def create_max_sliding_windows(df, window_size):
    data_values = df.drop(columns=['time']).values
    total_len = len(data_values)
    if total_len < window_size:
        raise ValueError(f"数据量不足")
    num_windows = total_len - window_size + 1
    shape = (num_windows, window_size, data_values.shape[1])
    strides = (data_values.strides[0], data_values.strides[0], data_values.strides[1])
    windows = np.lib.stride_tricks.as_strided(data_values, shape=shape, strides=strides)
    return windows, num_windows


def run(output_file="org_v1.npy", csv_file="org_v1.csv"):
    print(f"🚀 [Target: {SYMBOL}] 正在获取数据...")
    full_df = get_data()

    if full_df is not None:
        total_data_points = len(full_df)
        windows_data, actual_num = create_max_sliding_windows(full_df, WINDOW_SIZE)
        np.save(output_file, windows_data)
        full_df.to_csv(csv_file, index=False)
        print(f"✅ {SYMBOL} 数据准备就绪 | 窗口数: {actual_num} | 形状: {windows_data.shape}")
    else:
        print(f"❌ {SYMBOL} 数据获取失败。")
        sys.exit(1)  # 告知主程序失败


if __name__ == "__main__":
    run()