import MetaTrader5 as mt5
import time
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pyts.image import GramianAngularField
import pandas as pd

# 导入模型结构
from TPmodel import GafRegressionCNN
from Dmodel import GafCnnTransformer


# ==========================================
# 1. 实时特征引擎 (计算基础 f_0 到 f_8)
# ==========================================
class RealTimeFeatureEngine:
    def __init__(self, trend_weights, reg_weights, device="cpu", window_size=60):
        self.window_size = window_size
        self.device = device
        self.gaf_tool = GramianAngularField(image_size=window_size, method='summation', sample_range=(-1, 1))
        self.t_net = GafCnnTransformer(input_channels=12, output_dim=3).to(device).eval()
        self.r_net = GafRegressionCNN(input_channels=12, output_dim=6).to(device).eval()
        self.t_net.load_state_dict(torch.load(trend_weights, map_location=device))
        self.r_net.load_state_dict(torch.load(reg_weights, map_location=device))

    def get_latest_features(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, self.window_size + 120)
        if rates is None or len(rates) < self.window_size: return None, 0.0
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')[['open', 'high', 'low', 'close']]
        price = df['close'].iloc[-1]

        res_5m = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).ffill()
        res_15m = df.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).ffill()

        combined = pd.DataFrame(index=df.index)
        for prefix, data in [("1m", df), ("5m", res_5m), ("15m", res_15m)]:
            for col in ['open', 'high', 'low', 'close']:
                combined[f'{prefix}_{col}'] = data[col].reindex(df.index, method='ffill')

        raw_window = combined.dropna().tail(self.window_size).values.astype(np.float32)
        gaf_tensor = np.empty((12, self.window_size, self.window_size), dtype=np.float32)
        for ch in range(12):
            series = raw_window[:, ch].reshape(1, -1)
            s_min, s_max = series.min(), series.max()
            series_norm = 2 * (series - s_min) / (s_max - s_min + 1e-9) - 1
            gaf_tensor[ch, :, :] = self.gaf_tool.fit_transform(series_norm)[0]

        with torch.no_grad():
            gt = torch.from_numpy(gaf_tensor).unsqueeze(0).to(self.device)
            # 输出 f_0 到 f_8
            f_out = np.concatenate([self.t_net(gt).cpu().numpy().flatten(), self.r_net(gt).cpu().numpy().flatten()])
        return f_out, price


# ==========================================
# 2. 席位逻辑管理器 (完全对应 TripleSlotHardStopWrapper)
# ==========================================
class TripleSlotLiveManager:
    def __init__(self):
        # slots: [方向, 入场价, 止盈, 止损]
        self.slots = np.zeros((3, 4))
        self.last_actions = np.zeros(3)  # 对应训练代码中的 self.last_actions
        self.act_map = [-1, 0, 1]

    def update_and_predict(self, model, vec_norm, features, price):
        # 1. 构造 12 维反馈特征 (对应训练代码 step 中的 feedback 构建)
        feedback = []
        for i in range(3):
            s_dir = self.slots[i, 0]
            s_entry = self.slots[i, 1]
            s_tp = self.slots[i, 2]
            s_sl = self.slots[i, 3]

            pnl = (price / s_entry - 1) * s_dir if s_dir != 0 else 0
            dist_tp = (s_tp / price - 1) if s_dir != 0 else 0
            dist_sl = (s_sl / price - 1) if s_dir != 0 else 0
            feedback.extend([s_dir, pnl, dist_sl, dist_tp])

        # 2. 合并观测并获取预测
        obs = np.concatenate([features, feedback]).astype(np.float32)
        norm_obs = vec_norm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        current_raw_actions = action[0]  # [a0, a1, a2] (0,1,2)

        # 3. 核心逻辑处理 (对应 TripleSlotHardStopWrapper.step)
        for i in range(3):
            target_dir = self.act_map[current_raw_actions[i]]

            # --- A. 被动平仓检测 (硬止损) ---
            if self.slots[i, 0] != 0:
                s_dir, _, tp, sl = self.slots[i]
                hit_tp = (s_dir > 0 and price >= tp) or (s_dir < 0 and price <= tp)
                hit_sl = (s_dir > 0 and price <= sl) or (s_dir < 0 and price >= sl)
                if hit_tp or hit_sl:
                    self.slots[i] = 0  # 触发止盈止损，释放席位

            # --- B. 开仓逻辑 (席位空闲 & 意图开仓 & 上一步观望) ---
            if self.slots[i, 0] == 0 and target_dir != 0 and self.last_actions[i] == 0:
                # 使用 f_3-f_8 计算偏移
                high_off = features[3 + i * 2]
                low_off = features[4 + i * 2]

                self.slots[i, 0] = target_dir
                self.slots[i, 1] = price
                if target_dir > 0:  # 多
                    self.slots[i, 2], self.slots[i, 3] = price * (1 + high_off), price * (1 + low_off)
                else:  # 空
                    self.slots[i, 2], self.slots[i, 3] = price * (1 + low_off), price * (1 + high_off)

        # 更新动作历史
        self.last_actions = np.array([self.act_map[a] for a in current_raw_actions])
        return self.slots


# ==========================================
# 3. 运行主函数
# ==========================================
def run(SYMBOL="ETHUSDm", PPO_PATH="eth_ppo_v4", VEC_PATH="vec_eth_ppo_v4.pkl"):
    if not mt5.initialize(): return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine = RealTimeFeatureEngine("../CNN_Transformer/transformer_model_v1.pth", "../CNN/cnn_model_v1.pth", device)

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(21,))
            self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])

    ppo = PPO.load(PPO_PATH, device=device)
    vec_norm = VecNormalize.load(VEC_PATH, DummyVecEnv([lambda: DummyEnv()]))
    vec_norm.training = False

    manager = TripleSlotLiveManager()
    output_path = "D:/exnessMT5/MQL5/Files/ai_trade_signal.txt"

    print(f"🚀 策略引擎启动 | 交易品种: {SYMBOL}")

    while True:
        try:
            feats, price = engine.get_latest_features(SYMBOL)
            if feats is None: continue

            slots = manager.update_and_predict(ppo, vec_norm, feats, price)

            now_str = datetime.now().strftime('%H:%M:%S')

            # --- 核心修复：格式化数据防止输出 np.float ---
            ws = [int(slots[i, 0]) for i in range(3)]  # 转为整数
            tpsls = []
            for i in range(3):
                tpsls.append(round(float(slots[i, 2]), 2))  # TP
                tpsls.append(round(float(slots[i, 3]), 2))  # SL

            # 写入文件 (CSV格式)
            output_csv = f"{now_str},{ws[0]},{ws[1]},{ws[2]}," + ",".join(map(str, tpsls))
            with open(output_path, "w") as f:
                f.write(output_csv)

            # --- 核心修复：完整打印不省略 ---
            print(f"[{now_str}] 现价: {price:.2f} | 席位方向: {ws}")
            for i in range(3):
                status = "空闲" if ws[i] == 0 else ("多单" if ws[i] > 0 else "空单")
                if ws[i] != 0:
                    print(f"  └─ 席位 {i} [{status}]: 止盈={tpsls[i * 2]:.2f}, 止损={tpsls[i * 2 + 1]:.2f}")
                else:
                    print(f"  └─ 席位 {i} [空闲]")
            print("-" * 50)

        except Exception as e:
            print(f"❌ 运行错误: {e}")
        time.sleep(10)


if __name__ == "__main__":
    run()