import pandas as pd
import numpy as np
import torch
import time
import os
from pyts.image import GramianAngularField
from TPmodel import GafRegressionCNN  # 确保 TPmodel.py 在同目录下
import MetaTrader5 as mt5
# 定义存放数据的路径 (存放在 MT5 的公共文件夹或当前目录)
# 建议放在 MT5 的 Files 目录下，方便 MQL5 读取
# 通常路径为: C:/Users/你的用户名/AppData/Roaming/MetaQuotes/Terminal/XXX/MQL5/Files/ai_preds.txt
# 为了简单，先存放在当前脚本目录，你之后可以手动复制路径
# --- 配置 ---
SYMBOL = "ETHUSDm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_gaf_cnn_model.pth"

DATA_FILE = "D:/exnessMT5/MQL5/Files/ai_preds.txt"

def get_latest_data(window_size=60):
    """获取并对齐最新数据"""
    if not mt5.initialize():
        print("MT5 初始化失败")
        return None, None

    # 获取 200 条数据确保 15m 采样足够
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 200)
    if rates is None or len(rates) < window_size:
        return None, None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df_1m = df.set_index('time')

    # 对齐逻辑
    res_5m = df_1m['close'].resample('5min').ohlc().ffill()
    res_15m = df_1m['close'].resample('15min').ohlc().ffill()

    combined = pd.DataFrame(index=df_1m.index)
    for tf, data in [("1m", df_1m), ("5m", res_5m), ("15m", res_15m)]:
        for col in ['open', 'high', 'low', 'close']:
            combined[f'{tf}_{col}'] = data[col].reindex(df_1m.index, method='ffill')

    final_df = combined.dropna().tail(window_size)
    current_p = final_df['1m_close'].iloc[-1]
    return final_df, current_p


def predict(model, window_df):
    """特征转换与推理"""
    X_raw = window_df.values  # (60, 12)
    gaf = GramianAngularField(method='summation', sample_range=(-1, 1))
    X_gaf = np.empty((60, 60, 12), dtype=np.float32)

    for ch in range(12):
        series = X_raw[:, ch].reshape(1, -1)
        s_min, s_max = series.min(), series.max()
        # 局部归一化
        series_norm = (series - s_min) / (s_max - s_min + 1e-9) * 2 - 1
        X_gaf[:, :, ch] = gaf.fit_transform(series_norm)[0]

    # 转为 (1, 12, 60, 60) 张量
    tensor = torch.from_numpy(X_gaf).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        preds = model(tensor).cpu().numpy()[0]
    return preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GafRegressionCNN().to(device)
    model.load_state_dict(torch.load("best_gaf_cnn_model.pth", map_location=device))
    model.eval()

    print("AI 预测服务已启动，正在写入数据文件...")

    while True:
        df, current_p = get_latest_data()
        if df is not None:
            preds = predict(model, df)

            # 计算实际价格
            p1m_h, p1m_l = current_p * (1 + preds[0]), current_p * (1 + preds[1])
            p5m_h, p5m_l = current_p * (1 + preds[2]), current_p * (1 + preds[3])
            p15m_h, p15m_l = current_p * (1 + preds[4]), current_p * (1 + preds[5])

            # 将数据写入文件，格式：1m_H,1m_L,5m_H,5m_L,15m_H,15m_L
            try:
                with open(DATA_FILE, "w") as f:
                    f.write(f"{p1m_h},{p1m_l},{p5m_h},{p5m_l},{p15m_h},{p15m_l}")
                print(f"[{time.strftime('%H:%M:%S')}] 数据已更新到 {DATA_FILE}")
            except Exception as e:
                print(f"写入失败: {e}")

        time.sleep(6)


if __name__ == "__main__":
    main()