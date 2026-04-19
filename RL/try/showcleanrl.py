import os
import MetaTrader5 as mt5
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import pandas as pd
from pyts.image import GramianAngularField

# 假设您的模型定义文件在同目录下
from cleanrl import Agent
from TPmodel import GafRegressionCNN
from Dmodel import GafCnnTransformer


# ==========================================
# 1. 实时特征引擎 (强化预热版)
# ==========================================
class RealTimeFeatureEngine:
    def __init__(self, trend_weights, reg_weights, device="cpu", window_size=60):
        self.window_size = window_size
        self.device = device
        self.gaf_tool = GramianAngularField(image_size=window_size, method='summation', sample_range=(-1, 1))

        # 加载模型
        self.t_net = GafCnnTransformer(input_channels=12, output_dim=3).to(device).eval()
        self.r_net = GafRegressionCNN(input_channels=12, output_dim=6).to(device).eval()

        if os.path.exists(trend_weights) and os.path.exists(reg_weights):
            self.t_net.load_state_dict(torch.load(trend_weights, map_location=device))
            self.r_net.load_state_dict(torch.load(reg_weights, map_location=device))
            print("✅ 基础卷积特征网络加载成功")
        else:
            print("⚠️ 警告：未找到特征网络权重文件，将使用随机初始化进行测试")

    def _internal_inference(self, r1, r5, r15):
        """将 3 个周期的原始数据转换为 9 维特征向量 (f0-f8)"""
        # 提取并组合 12 通道数据 (1m OHLD, 5m OHLD, 15m OHLD)
        cols = ['open', 'high', 'low', 'close']
        d1 = pd.DataFrame(r1)[cols].values
        d5 = pd.DataFrame(r5)[cols].values
        d15 = pd.DataFrame(r15)[cols].values

        # 水平堆叠成 (60, 12)
        combined = np.hstack([d1, d5, d15]).astype(np.float32)

        # 生成 GAF 张量 (12, 60, 60)
        gaf_tensor = np.empty((12, self.window_size, self.window_size), dtype=np.float32)
        for ch in range(12):
            series = combined[:, ch].reshape(1, -1)
            s_min, s_max = series.min(), series.max()
            series_norm = 2 * (series - s_min) / (s_max - s_min + 1e-9) - 1
            gaf_tensor[ch, :, :] = self.gaf_tool.fit_transform(series_norm)[0]

        with torch.no_grad():
            gt = torch.from_numpy(gaf_tensor).unsqueeze(0).to(self.device)
            f_out = np.concatenate([
                self.t_net(gt).cpu().numpy().flatten(),
                self.r_net(gt).cpu().numpy().flatten()
            ])
        return f_out

    def get_latest_features(self, symbol):
        """获取当前最新的分钟特征，直接对 numpy 数组进行点差补偿"""
        # 1. 获取数据
        r1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 60)
        r5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 60)
        r15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 60)

        # 2. 获取点差信息
        info = mt5.symbol_info(symbol)
        if r1 is None or r5 is None or r15 is None or info is None or len(r1) < 60:
            return None, 0.0

        # 计算实际价格点差 (info.spread 是点数 * 最小步长)
        current_spread = info.spread * info.point

        # 3. 核心修改：直接在 numpy 结构化数组的副本上修改 high 值
        # 注意：使用 .copy() 是为了不破坏 MT5 缓存的原始数据
        r1_adj = r1.copy()
        r5_adj = r5.copy()
        r15_adj = r15.copy()

        r1_adj['high'] += current_spread
        r5_adj['high'] += current_spread
        r15_adj['high'] += current_spread

        # 4. 传入推理逻辑
        f_out = self._internal_inference(r1_adj, r5_adj, r15_adj)
        return f_out, r1[-1]['close']

    def get_warmup_sequence(self, symbol, count=60):
        """一次性回溯获取 60 个步长的数据 (总计需回溯约 120 根 K 线)"""
        print(f"\n⏳ 开始从 MT5 获取 {symbol} 历史数据进行预热...")

        # 获取足够的历史深度 (当前位置往回数 120 根)
        total_len = count + self.window_size
        all_m1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, total_len)
        all_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, total_len)
        all_m15 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, total_len)

        if all_m1 is None or len(all_m1) < total_len:
            print("❌ 错误：无法获取足够的历史数据，请检查品种名称或 MT5 连接。")
            return None, None

        seq_feats, seq_prices = [], []

        print("-" * 75)
        print(f"{'序号':<5} | {'窗口开始时间':<18} | {'窗口结束时间':<18} | {'收盘价':<10}")
        print("-" * 75)

        # 本地滑动切片计算
        for i in range(count):
            # 从最老的数据向最新的数据推进
            start = i
            end = i + self.window_size

            r1_s = all_m1[start:end]
            r5_s = all_m5[start:end]
            r15_s = all_m15[start:end]

            f = self._internal_inference(r1_s, r5_s, r15_s)
            p = r1_s[-1]['close']

            seq_feats.append(f)
            seq_prices.append(p)

            # 打印每一步的窗口范围
            t_open = datetime.fromtimestamp(r1_s[0]['time']).strftime('%Y-%m-%d %H:%M')
            t_close = datetime.fromtimestamp(r1_s[-1]['time']).strftime('%Y-%m-%d %H:%M')
            if i % 10 == 0 or i == count - 1:  # 抽样打印或打印最后一条
                print(f"{i + 1:<5} | {t_open:<18} | {t_close:<18} | {p:<10.2f}")

        print("-" * 75)
        return seq_feats, seq_prices


# ==========================================
# 2. 席位逻辑管理器
# ==========================================
class TripleSlotLiveManager:
    def __init__(self, window_size=60):
        self.slots = np.zeros((3, 4))  # [dir, entry, tp, sl]
        self.last_actions = np.zeros(3)
        self.window_size = window_size
        self.obs_buffer = []

    def update_and_predict(self, agent, device, features, price, is_warmup=False):
        # 1. 构造反馈特征 (12维)
        feedback = []
        for i in range(3):
            s_dir, s_entry, s_tp, s_sl = self.slots[i]
            pnl = (price / s_entry - 1) * s_dir if s_dir != 0 else 0
            dist_tp = (s_tp / price - 1) if s_dir != 0 else 0
            dist_sl = (s_sl / price - 1) if s_dir != 0 else 0
            feedback.extend([s_dir, pnl, dist_sl, dist_tp])

        # 2. 合并 21 维观测并维护窗口
        current_step_obs = np.concatenate([features, feedback]).astype(np.float32)
        self.obs_buffer.append(current_step_obs)
        if len(self.obs_buffer) > self.window_size:
            self.obs_buffer.pop(0)

        # 预热期不进行交易决策
        if is_warmup or len(self.obs_buffer) < self.window_size:
            return self.slots

        # 3. 神经网络预测
        with torch.no_grad():
            input_obs = torch.as_tensor(np.array(self.obs_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            _, _, _, _, weights = agent.get_action_and_value(input_obs)
            current_weights = weights.cpu().numpy()[0]

        # 4. 执行席位更新逻辑 (基于训练时的 HardStop 规则)
        for i in range(3):
            target_dir = current_weights[i]

            # 平仓检测
            if self.slots[i, 0] != 0:
                s_dir, _, tp, sl = self.slots[i]
                hit_tp = (s_dir > 0 and price >= tp) or (s_dir < 0 and price <= tp)
                hit_sl = (s_dir > 0 and price <= sl) or (s_dir < 0 and price >= sl)
                if hit_tp or hit_sl:
                    self.slots[i] = 0

            # 开仓逻辑
            if self.slots[i, 0] == 0 and target_dir != 0 and self.last_actions[i] == 0:
                # 获取该时刻网络预测的 TP/SL 偏移
                h_off, l_off = features[3 + i * 2], features[4 + i * 2]
                self.slots[i, 0] = target_dir
                self.slots[i, 1] = price
                if target_dir > 0:
                    self.slots[i, 2], self.slots[i, 3] = price * (1 + h_off), price * (1 + l_off)
                else:
                    self.slots[i, 2], self.slots[i, 3] = price * (1 + l_off), price * (1 + h_off)

        self.last_actions = current_weights.copy()
        return self.slots


# ==========================================
# 3. 运行主函数
# ==========================================
def run(SYMBOL="ETHUSDm", MODEL_PATH="eth_ppo_cleanrl_v1.pt"):
    if not mt5.initialize():
        print("MT5 初始化失败")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化引擎与席位管理器
    engine = RealTimeFeatureEngine(
        "../CNN_Transformer/transformer_model_v1.pth",
        "../CNN/cnn_model_v1.pth",
        device
    )
    manager = TripleSlotLiveManager(window_size=60)

    # 2. 构造 Agent 环境容器
    class MockSpace:
        def __init__(self, shape, nvec=None):
            self.shape, self.nvec = shape, nvec

    class DummyEnvContainer:
        def __init__(self):
            self.single_observation_space = MockSpace(shape=(60, 21))
            self.single_action_space = MockSpace(shape=None, nvec=[3, 3, 3])
            self.num_envs = 1

    agent = Agent(DummyEnvContainer(), features_dim=256).to(device)

    # 3. 加载训练权重
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'means' in checkpoint: agent.means.copy_(checkpoint['means'])
            if 'stds' in checkpoint: agent.stds.copy_(checkpoint['stds'])
            print(f"✅ 成功加载 Checkpoint 权重。")
        else:
            agent.load_state_dict(checkpoint, strict=False)
    agent.eval()

    # 4. 执行深度预热 (秒级补齐 60 分钟记忆)
    hist_feats, hist_prices = engine.get_warmup_sequence(SYMBOL, count=60)
    if hist_feats:
        for f, p in zip(hist_feats, hist_prices):
            manager.update_and_predict(agent, device, f, p, is_warmup=True)
        print(f"✅ 预热补齐完成，Buffer 长度: {len(manager.obs_buffer)}")
    else:
        print("❌ 预热失败，程序退出。")
        return

    # 5. 进入实盘循环
    output_path = "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files/ai_trade_signal.txt"
    last_min = -1
    print(f"\n🚀 实盘监控已开启 [{SYMBOL}]...")

    while True:
        try:
            now = datetime.now()
            # 严格对齐每分钟的开始
            if now.minute != last_min:
                time.sleep(1)  # 稍等 1 秒确保服务器 K 线数据刷新

                feats, price = engine.get_latest_features(SYMBOL)
                if feats is not None:
                    slots = manager.update_and_predict(agent, device, feats, price)

                    # 格式化信号
                    ws = [int(slots[i, 0]) for i in range(3)]
                    tpsls = []
                    for i in range(3):
                        tpsls.append(round(float(slots[i, 2]), 5))  # TP
                        tpsls.append(round(float(slots[i, 3]), 5))  # SL

                    output_content = f"{now.strftime('%H:%M:%S')},{ws[0]},{ws[1]},{ws[2]}," + ",".join(map(str, tpsls))

                    # 写入文件供 EA 读取
                    with open(output_path, "w") as f:
                        f.write(output_content)

                    last_min = now.minute
                    print(f"[{now.strftime('%H:%M:%S')}] 价格: {price:.2f} | 席位: {ws} |tp/sl: {tpsls}")

        except Exception as e:
            print(f"❌ 运行异常: {e}")

        time.sleep(1)  # 高频检查分钟切换


if __name__ == "__main__":
    run(SYMBOL="ETHUSDm")