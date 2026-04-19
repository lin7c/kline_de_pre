import os
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import gym_trading_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# ==========================================
# 1. 优化后的特征提取器 (滑动窗口 + 仓位反馈)
# ==========================================
class OptimizedFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, feature_cols, window_size=10):
        super().__init__(env)
        self.feature_cols = feature_cols
        self.window_size = window_size

        # 预取特征数据
        self.raw_features = self.env.unwrapped.df[self.feature_cols].values.astype(np.float32)

        # 观测空间: (窗口大小 * 15个特征) + 1个当前仓位
        self.num_features = len(feature_cols)
        self.obs_shape = (self.window_size * self.num_features + 1,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        env_uw = self.env.unwrapped
        idx = getattr(env_uw, '_step', 0)

        # 1. 提取滑动窗口
        start_idx = max(0, idx - self.window_size + 1)
        window_data = self.raw_features[start_idx: idx + 1]

        if len(window_data) < self.window_size:
            padding = np.tile(self.raw_features[0], (self.window_size - len(window_data), 1))
            window_data = np.vstack([padding, window_data])

        # 2. 获取当前仓位 (解决 AttributeError)
        try:
            # 使用官方建议的索引方式获取最近一次仓位
            current_pos_val = env_uw.historical_info['position', -1]
        except (AttributeError, KeyError, IndexError):
            current_pos_val = 0.0

        current_position = np.array([float(current_pos_val)], dtype=np.float32)
        return np.concatenate([window_data.flatten(), current_position])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ==========================================
# 2. 修正后的夏普率奖励函数 (基于净值变化)
# ==========================================
def sharpe_reward(history):
    lookback = 30
    # 获取最近的投资组合估值 (Portfolio Valuation)
    try:
        valuations = history["portfolio_valuation", -(lookback + 1):]
    except (KeyError, IndexError):
        return 0.0

    if len(valuations) < 2:
        return 0.0

    # 计算收益率: (当前估值 / 上一次估值) - 1
    # 增加 epsilon 防止除以 0
    returns = np.diff(valuations) / (valuations[:-1] + 1e-7)

    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-6

    # 夏普比率作为奖励 (鼓励高收益的同时惩罚高波动)
    return mean_return / std_return


# ==========================================
# 3. 环境构造
# ==========================================
def make_env(rank, seed, df_data, cols):
    def _init():
        base_env = gym.make('TradingEnv',
                            df=df_data,
                            trading_fees=0.0005,
                            positions=[-1, 0, 1],
                            reward_function=sharpe_reward,
                            portfolio_initial_value=1.0)

        env = OptimizedFeatureWrapper(base_env, cols, window_size=10)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


# ==========================================
# 4. 训练执行
# ==========================================
def run(TRAIN_DATA_PATH="ppo_x_v1.csv", MODEL_PATH="eth_ppo_v4_sharpe"):
    DEVICE = "cpu"
    VEC_PATH = f"vec_{MODEL_PATH}.pkl"
    MODEL_FILE = f"{MODEL_PATH}.zip"

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ 错误: 找不到文件 {TRAIN_DATA_PATH}")
        return

    # 数据预处理
    df = pd.read_csv(TRAIN_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    feature_columns = [f"f_{i}" for i in range(15)]

    # 并行环境设置 (Windows下建议先设为4-6观察稳定性)
    num_cpu = 6
    venv = SubprocVecEnv([make_env(i, 42, df, feature_columns) for i in range(num_cpu)])

    # 归一化
    if os.path.exists(VEC_PATH):
        print(f">>> 加载归一化配置: {VEC_PATH}")
        venv = VecNormalize.load(VEC_PATH, venv)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

    # 模型配置
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if os.path.exists(MODEL_FILE):
        print(f">>> 载入现有模型: {MODEL_FILE}")
        model = PPO.load(MODEL_FILE, env=venv, device=DEVICE)
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            device=DEVICE,
            learning_rate=3e-5,  # 略微调低学习率以应对夏普奖励的波动
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./logs/tb_logs/"
        )

    # 回调
    eval_callback = EvalCallback(venv, best_model_save_path=f"./best_{MODEL_PATH}",
                                 log_path="../logs/", eval_freq=5000)

    try:
        print(">>> 开始训练...")
        model.learn(total_timesteps=5000000, callback=eval_callback)
    except KeyboardInterrupt:
        print(">>> 手动停止")
    finally:
        model.save(MODEL_PATH)
        venv.save(VEC_PATH)
        print(">>> 模型与归一化参数已保存")


if __name__ == "__main__":
    os.makedirs("../logs/tb_logs", exist_ok=True)
    run()