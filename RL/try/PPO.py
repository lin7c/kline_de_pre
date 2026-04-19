import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env
from pyts.image import GramianAngularField
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import FrameStackObservation
from mini import sharpe_reward

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, 60, d_model]
        return x + self.pe[:, :x.size(1)]


class TradingDecoderExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # 此时 observation_space.shape 为 (60, 27)
        super().__init__(observation_space, features_dim)

        self.d_model = 256
        self.max_seq_len = 60

        # 1. 统一线性映射：直接处理全部 27 维特征
        self.input_proj = nn.Linear(27, self.d_model)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len)

        # 3. Transformer Decoder 结构
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

    def forward(self, observations):
        # observations shape: [Batch, 60, 27]

        # 第一步：将 27 维特征投影到 d_model 维度
        # shape: [Batch, 60, 256]
        x = self.input_proj(observations)

        # 第二步：加入位置信息
        x = self.pos_encoder(x)

        # 第三步：准备 Query 和 Memory
        # Memory (m): 整个 60 步的序列 [Batch, 60, 256]
        # Query (q): 取序列的最后一步（当前时刻），并保持维度 [Batch, 1, 256]
        m = x
        q = x[:, -1:, :]

        # 第四步：解码器计算
        # 让当前的特征 (q) 去注意力机制里“回看”过去 60 步 (m)
        decoded = self.transformer_decoder(q, m)

        # 返回形状 [Batch, 256] 传给 Actor/Critic
        return decoded.squeeze(1)

# ==========================================
# Wrapper
# ==========================================
class TripleSlotHardStopWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 动作空间：3个席位，每席位 [空, 观望, 多]
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])
        self.act_map = [-1, 0, 1]

        # slots 存储: [方向, 入场价, 止盈价, 止损价]
        self.slots = np.zeros((3, 4))

        orig_shape = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape + 12,), dtype=np.float32
        )
        self.last_actions = np.zeros(3)

    def step(self, action):
        # 1. 获取基础环境及当前价格信息
        env_uw = self.env.unwrapped
        # 确保索引不越界
        current_idx = min(getattr(env_uw, '_step', 0), len(env_uw.df) - 1)
        row = env_uw.df.iloc[current_idx]
        # 提取关键价格点
        price = row['close']
        price_high = row['high']
        price_low = row['low']

        total_step_reward = 0.0

        # 2. 遍历三个席位
        for i in range(3):
            target_dir = self.act_map[action[i]]
            current_dir = self.slots[i, 0]
            entry_price = self.slots[i, 1]

            # --- A. 被动平仓检测 (使用 High/Low 增强准确性) ---
            if current_dir != 0:
                tp_price = self.slots[i, 2]
                sl_price = self.slots[i, 3]

                executed_price = None  # 记录实际成交价

                if current_dir > 0:  # 多头持仓
                    if price_low <= sl_price:  # 先判止损（保守原则）
                        executed_price = sl_price
                    elif price_high >= tp_price:  # 后判止盈
                        executed_price = tp_price
                else:  # 空头持仓
                    if price_high >= sl_price:  # 先判止损
                        executed_price = sl_price
                    elif price_low <= tp_price:  # 后判止盈
                        executed_price = tp_price

                if executed_price is not None:
                    # 计算基于触发价的收益（更符合真实逻辑）
                    total_step_reward += (executed_price / entry_price - 1.0) * current_dir
                    self.slots[i] = 0
                    current_dir = 0

            # --- B. 开仓逻辑 (仅在席位空闲时接受新动作) ---
            # 条件：1.当前没持仓 2.模型想开仓 3.上一步是观望(防止连续开仓/刷单)
            if current_dir == 0 and target_dir != 0 and self.last_actions[i] == 0:
                # 动态获取数据列：席位0(f3,f4), 席位1(f5,f6), 席位2(f7,f8)
                high_offset = row[f"f_{3 + i * 2}"]  # 高点偏移量 (如 0.002)
                low_offset = row[f"f_{4 + i * 2}"]  # 低点偏移量 (如 -0.005)

                self.slots[i, 0] = target_dir  # 方向
                self.slots[i, 1] = price  # 入场价

                if target_dir > 0:  # 做多：止盈看高点，止损看低点
                    self.slots[i, 2] = price * (1 + high_offset)
                    self.slots[i, 3] = price * (1 + low_offset)
                else:  # 做空：止盈看低点，止损看高点
                    self.slots[i, 2] = price * (1 + low_offset)
                    self.slots[i, 3] = price * (1 + high_offset)

        # 3. 更新上一次动作记录（用于逻辑判定）
        self.last_actions = np.array([self.act_map[a] for a in action])

        # 4. 计算总仓位并同步到底层环境
        # 即使模型这一步输出“平仓”，如果没触发TP/SL，slots不会变，net_position就不会变
        net_position = np.sum(self.slots[:, 0]) / 3.0
        pos_idx = int(np.argmin(np.abs(np.array(env_uw.positions) - net_position)))

        # 调用底层 step (这里会处理手续费、净值更新等)
        obs, _, term, trunc, info = self.env.step(pos_idx)

        # 5. 封装监控数据 (用于 Callback 记录到 Tensorboard)
        monitor_slots = []
        for i in range(3):
            s_dir = self.slots[i, 0]
            s_entry = self.slots[i, 1]
            s_tp = self.slots[i, 2]
            s_sl = self.slots[i, 3]

            monitor_slots.append({
                "dir": s_dir,
                "entry": s_entry,
                "tp": s_tp,
                "sl": s_sl,
                "price": price,  # 当前价格
                "dist_tp": (s_tp / price - 1) if s_dir != 0 else 0,
                "dist_sl": (s_sl / price - 1) if s_dir != 0 else 0,
                "pnl": (price / s_entry - 1) * s_dir if s_dir != 0 else 0
            })

        info["monitor_slots"] = monitor_slots
        info["monitor_total_pos"] = net_position

        # 6. 构建反馈给模型的 Observation (12维状态反馈)
        feedback = []
        for s in monitor_slots:
            # 这里的特征有助于模型感知距离止盈止损还有多远
            feedback.extend([s["dir"], s["pnl"], s["dist_sl"], s["dist_tp"]])

        return np.concatenate([obs, feedback]).astype(np.float32), total_step_reward, term, trunc, info

    def reset(self, **kwargs):
        self.slots = np.zeros((3, 4))
        self.last_actions = np.zeros(3)
        obs, info = self.env.reset(**kwargs)  # 这里的 obs 是 15 维
        return np.concatenate([obs, np.zeros(12, dtype=np.float32)]), info
from stable_baselines3.common.callbacks import BaseCallback


class TradingMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingMonitorCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 从 VecEnv 提取 info 字典
        infos = self.locals.get("infos")
        # 提取模型当前步的原始动作输出 (MultiDiscrete: [a1, a2, a3])
        actions = self.locals.get("actions")

        if infos and actions is not None:
            info = infos[0]
            action = actions[0]  # 获取第一个环境实例的动作

            # 1. 监控模型原始 Action 输出 (映射回 -1, 0, 1)
            # 这有助于观察模型是否在“撞墙”（一直想平仓但被规则禁止）
            act_map = [-1, 0, 1]
            self.logger.record("action_raw/slot_0", act_map[action[0]])
            self.logger.record("action_raw/slot_1", act_map[action[1]])
            self.logger.record("action_raw/slot_2", act_map[action[2]])

            # 2. 监控实际席位状态 (Current Position Status)
            slots_data = info.get("monitor_slots", [])
            for i, slot in enumerate(slots_data):
                prefix = f"slot_status_{i}/"
                # 当前实际方向 (1, 0, -1)
                self.logger.record(f"{prefix}current_dir", slot.get("dir", 0))
                # 距离止盈止损的距离 (百分比)
                self.logger.record(f"{prefix}dist_tp", slot.get("dist_tp", 0))
                self.logger.record(f"{prefix}dist_sl", slot.get("dist_sl", 0))
                # 当前席位的浮动盈亏 (仅监控用)
                self.logger.record(f"{prefix}floating_pnl", slot.get("pnl", 0))

            # 3. 总体指标
            self.logger.record("custom/net_exposure", info.get("monitor_total_pos", 0))
            # 记录实际结算奖励 (只有平仓那一刻不为0)
            self.logger.record("custom/settled_reward", self.locals.get("rewards")[0])

        return True
class FeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, cols):
        super().__init__(env)
        self.cols = cols
        # 预先转为 Numpy，避免每步都做 df[cols] 操作
        self.data_pool = self.env.unwrapped.df[self.cols].values.astype(np.float32)
        # 观测空间改为 (27,) 也就是单行特征
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(cols),), dtype=np.float32
        )

    def observation(self, obs):
        env_uw = self.env.unwrapped
        idx = getattr(env_uw, '_step', 0)
        # 只返回当前这一行数据
        return self.data_pool[min(idx, len(self.data_pool) - 1)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 强制返回全 0 的初始特征，而不是第一行数据的重复
        zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return zero_obs, info

def make_env(rank, seed=0, df_data=None, cols=None):
    def _init():
        # 传入预处理好的 numpy 数组，彻底干掉 Pandas 瓶颈
        base_env = gym.make('TradingEnv',
                            df=df_data,
                            trading_fees=0.0005,
                            positions=np.around(np.linspace(-1, 1, 7), 1).tolist(),
                            portfolio_initial_value=1,
                            reward_function=sharpe_reward,
                            windows=None)
        # 嵌套你的包装器
        env = FeatureWrapper(base_env, cols)
        env = TripleSlotHardStopWrapper(env)
        env = FrameStackObservation(env, 60)
        env.reset(seed=seed + rank)
        return env

    return _init
# ==========================================
# 训练主函数
# ==========================================
def run(TRAIN_DATA_PATH="ppo_x_v1.csv", MODEL_PATH="ppo_model_v4"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VEC_PATH = f"vec_{MODEL_PATH}.pkl"
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ 找不到训练数据: {TRAIN_DATA_PATH}")
        # 如果文件不存在，尝试调用你之前的 export_training_data 逻辑
        return

    # 1. 加载数据
    df = pd.read_csv(TRAIN_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 自动识别特征列 f_0 到 f_8
    feature_columns = [f"f_{i}" for i in range(15)]

    # 2. 创建基础环境
    # 注意：positions 需要覆盖 [-1, 1] 范围以匹配 MultiLevelHardStopWrapper 的输出
    fine_positions = np.around(np.linspace(-1, 1, 7), 1).tolist()
    base_env = gym.make('TradingEnv',
                        df=df,
                        trading_fees=0.0005,
                        positions=fine_positions,
                        portfolio_initial_value=1000,

                        windows=None)

    # 3. 嵌套包装器
    # 先提取特征，再套用硬性约束逻辑
    env = FeatureWrapper(base_env, feature_columns)
    env = TripleSlotHardStopWrapper(env)

    # 4. 随机动作测试循环 (验证逻辑是否通畅)
    # 4. 随机动作测试循环 (验证三席位独立管理逻辑)
    # 4. 随机动作测试循环 (验证三席位独立管理逻辑)
    obs, _ = env.reset()
    print(f"%n>>> 初始观测维度: {len(obs)}")
    print(">>> 启动三席位随机步进测试 (显示当前价、入场价、止盈/止损)...")

    for i in range(20):
        # 采样动作 [a1, a2, a3]
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)

        # 提取 Wrapper 中封装的详细监控数据
        slots = info.get("monitor_slots", [])
        net_pos = info.get("monitor_total_pos", 0)
        net_worth = info.get('portfolio_valuation', 0)

        # 映射 Action 动作为可读符号
        act_map_desc = {0: "做空", 1: "观望", 2: "做多"}
        act_str = f"[{act_map_desc[action[0]]}|{act_map_desc[action[1]]}|{act_map_desc[action[2]]}]"

        # --- 打印本步概览 ---
        print("-" * 110)
        print(f"Step {i:02d} | 动作意图:{act_str} | 总仓位:{net_pos:>5.2f} | 奖励:{r:>8.5f} | 净值:{net_worth:>8.2f}")

        # --- 打印每个席位的详细价格逻辑 ---
        for idx, s in enumerate(slots):
            if s['dir'] != 0:
                side = "🔴多单" if s['dir'] > 0 else "🟢空单"
                # 打印：席位状态 | 当前价 | 入场价 | 止盈价 | 止损价 | 浮动盈亏
                print(f"  Slot {idx}: {side} | 现价:{s['price']:>8.2f} | 入场:{s['entry']:>8.2f} | "
                      f"止盈:{s['tp']:>8.2f} ({s['dist_tp'] * 100:>+5.2f}%) | "
                      f"止损:{s['sl']:>8.2f} ({s['dist_sl'] * 100:>+5.2f}%) | PnL:{s['pnl'] * 100:>+6.3f}%")
            else:
                print(f"  Slot {idx}: ⚪空闲")

        # 如果本步有结算（TP/SL 触发）
        if r != 0:
            print(f"      >>> 💰 [结算触发] 本步获得已实现收益: {r:.6f}")

        if term or trunc:
            print(">>> Episode 结束，重置环境")
            obs, _ = env.reset()

    print("%n" + "=" * 50)
    print(">>> 测试结束，准备开始正式训练...")
    print("=" * 50 + "%n")

    # 5. 向量化与归一化
    env = Monitor(env)
    # 2. 开启多进程环境 (建议设置为 CPU 核心数的一半或相等)
    num_cpu = 24
    venv = SubprocVecEnv([make_env(i, df_data=df, cols=feature_columns) for i in range(num_cpu)])
    if os.path.exists(VEC_PATH):
        print(f">>> 正在加载归一化统计数据: {VEC_PATH}")
        # 加载已有的归一化参数
        venv = VecNormalize.load(VEC_PATH, venv)
    else:
        print(">>> 创建新的归一化环境")
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)
    # 6. 回调函数配置
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30,  # 连续 30 次评估无改善则停止
        min_evals=20,
        verbose=1
    )

    eval_callback = EvalCallback(
        venv,
        best_model_save_path=f"./logs/best_{MODEL_PATH}",
        log_path="../logs/",
        eval_freq=10000,  # 每 1w 步评估一次
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=stop_train_callback
    )
    # 实例化监控 Callback
    trading_monitor = TradingMonitorCallback()

    # 将其与原有的 eval_callback 组合
    # CallbackList 会按顺序执行它们
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([eval_callback, trading_monitor])
    # 7. 定义 PPO 模型
    # 针对“时间衰减奖励”这种具有一定挑战性的奖励函数，微调了网络深度和熵系数
    MODEL_FILE = f"{MODEL_PATH}.zip"
    policy_kwargs = dict(
        features_extractor_class=TradingDecoderExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        # 这里的 net_arch 是在 Transformer 输出之后接的 MLP 层
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )
    if os.path.exists(MODEL_FILE):
        print(f">>> 找到现有模型 {MODEL_FILE}，正在加载以继续训练...")
        # 加载模型并关联当前环境
        model = PPO.load(MODEL_FILE, env=venv, device='cpu')
    else:
        print(">>> 未找到现有模型，正在初始化新模型...")
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            device='cpu',
            learning_rate=1e-4,
            batch_size=128,
            n_steps=8192,
            n_epochs=10,
            ent_coef=0.05,
            gamma=0.999,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/tb_logs/"
        )

    # 8. 执行训练
    try:
        model.learn(
            total_timesteps=5000000,
            callback=callback,
            tb_log_name=MODEL_PATH,
            reset_num_timesteps=False  # 设为 False 以衔接之前的训练步数
        )
    except KeyboardInterrupt:
        print(">>> 训练被人为中断，正在保存当前进度...")

    # 9. 保存结果
    model.save(MODEL_PATH)
    venv.save(VEC_PATH)
    print(f">>> 训练任务结束。模型：{MODEL_PATH}，归一化文件：{VEC_PATH}")


if __name__ == "__main__":
    # 确保日志目录存在
    os.makedirs("../logs/tb_logs", exist_ok=True)
    run(MODEL_PATH="eth_ppo_v4")