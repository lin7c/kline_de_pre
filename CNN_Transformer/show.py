import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import time
from pyts.image import GramianAngularField
from Dmodel import GafCnnTransformer  # 确保使用的是适配 NHWC 和回归输出的版本

# --- 配置 ---
SYMBOL = "ETHUSDm"
# 建议更新权重文件名以区分回归模型
MODEL_PATH = "../RL/best_trend_regressor.pth"
DATA_FILE = "D:/exnessMT5/MQL5/Files/ai_probs.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latest_data(window_size=60):
    """获取并对齐最新数据"""
    if not mt5.initialize():
        return None, None

    # 15m 采样 60 个点需要至少 900 分钟数据，这里取 1000 条确保覆盖
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 1000)
    if rates is None or len(rates) < window_size:
        return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df_1m = df.set_index('time')

    # 对齐逻辑 (OHLC)
    res_5m = df_1m['close'].resample('5min').ohlc().ffill()
    res_15m = df_1m['close'].resample('15min').ohlc().ffill()

    combined = pd.DataFrame(index=df_1m.index)
    # 构建 12 个通道 (1m_OHLC, 5m_OHLC, 15m_OHLC)
    for tf, data in [("1m", df_1m), ("5m", res_5m), ("15m", res_15m)]:
        for col in ['open', 'high', 'low', 'close']:
            combined[f'{tf}_{col}'] = data[col].reindex(df_1m.index, method='ffill')

    final_df = combined.dropna().tail(window_size)
    current_p = final_df['1m_close'].iloc[-1]
    return final_df, current_p


def get_live_gaf_tensor(window_size=60):
    """获取数据并转换为 NHWC 格式 (1, 60, 60, 12)"""
    df, curr_p = get_latest_data(window_size)
    if df is None: return None, None

    X_raw = df.values  # (60, 12)
    gaf = GramianAngularField(method='summation')

    # --- 核心修改：构建 NHWC 容器 (Batch, H, W, C) ---
    X_gaf = np.empty((1, 60, 60, 12), dtype=np.float32)

    for ch in range(12):
        series = X_raw[:, ch].reshape(1, -1)
        # 归一化至 [-1, 1] 以适配 GAF
        s_min, s_max = series.min(), series.max()
        if s_max - s_min < 1e-9:
            series_norm = np.zeros_like(series)
        else:
            series_norm = (series - s_min) / (s_max - s_min) * 2 - 1

        # 将生成的 (60, 60) 填入最后一维 (Channel)
        X_gaf[0, :, :, ch] = gaf.fit_transform(series_norm)[0]

    return torch.from_numpy(X_gaf).to(DEVICE), curr_p


def main():
    if not mt5.initialize():
        print("MT5 初始化失败")
        return

    # 初始化模型（内部会自动处理 NHWC -> NCHW 转换）
    model = GafCnnTransformer(input_channels=12, output_dim=3).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"成功加载回归模型权重: {MODEL_PATH}")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    model.eval()
    print("AI 趋势动能预测服务已启动 (输出为强度得分，正数看涨，负数看跌)...")

    while True:
        tensor, curr_p = get_live_gaf_tensor()
        if tensor is not None:
            with torch.no_grad():
                # 现在的输出是 Score，范围大约在 [-2.0, 2.0]
                scores = model(tensor).cpu().numpy()[0]

                # 写入文件供 EA 使用
            try:
                with open(DATA_FILE, "w") as f:
                    # 格式：1m_score, 5m_score, 15m_score, current_price
                    f.write(f"{scores[0]:.4f},{scores[1]:.4f},{scores[2]:.4f},{curr_p:.2f}")

                # 控制台打印：更直观地显示动能方向
                m1_dir = "↑" if scores[0] > 0 else "↓"
                m5_dir = "↑" if scores[1] > 0 else "↓"
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"1m: {m1_dir} {scores[0]:.3f} | "
                      f"5m: {m5_dir} {scores[1]:.3f} | "
                      f"15m: {scores[2]:.3f} | 价格: {curr_p}")
            except Exception as e:
                print(f"写入错误: {e}")

        time.sleep(1)  # 每10秒预测一次


if __name__ == "__main__":
    main()