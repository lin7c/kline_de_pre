import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import gymnasium as gym
import gym_trading_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ==========================================
# 1. 核心适配：必须与训练时的 Wrapper 完全一致
# ==========================================
class OptimizedFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, feature_cols, window_size=10):
        super().__init__(env)
        self.feature_cols = feature_cols
        self.window_size = window_size
        # 预取特征
        self.raw_features = self.env.unwrapped.df[self.feature_cols].values.astype(np.float32)

        # 观测空间: (窗口 * 特征数) + 1个仓位
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

        # 2. 获取当前仓位 (从历史信息中提取)
        try:
            current_pos_val = env_uw.historical_info['position', -1]
        except (AttributeError, KeyError, IndexError):
            current_pos_val = 0.0

        current_position = np.array([float(current_pos_val)], dtype=np.float32)
        return np.concatenate([window_data.flatten(), current_position])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


# ==========================================
# 2. 交互式可视化函数
# ==========================================
def interactive_visualize(model_path, vec_path, train_data_path):
    # 加载数据
    df = pd.read_csv(train_data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    feature_columns = [f"f_{i}" for i in range(9)]

    # 创建环境
    base_env = gym.make('TradingEnv',
                        df=df,
                        positions=[-1, 0, 1],
                        trading_fees=0.0005,
                        portfolio_initial_value=1.0)

    # 使用最新的 Wrapper
    env = OptimizedFeatureWrapper(base_env, feature_columns, window_size=10)
    venv = DummyVecEnv([lambda: env])

    # 加载归一化配置
    if os.path.exists(vec_path):
        print(f">>> 加载归一化文件: {vec_path}")
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False
        venv.norm_reward = False

    # 加载模型
    model = PPO.load(model_path, env=venv, device='cpu')

    # --- 预运行收集回测数据 ---
    print("正在根据模型策略生成交易路径...")
    history_records = []
    obs = venv.reset()

    # 遍历整个 DataFrame 长度
    for step_idx in range(len(df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = venv.step(action)

        info = infos[0]
        env_uw = env.unwrapped
        # 获取当前步骤在 DataFrame 中的实际索引
        curr_ptr = getattr(env_uw, '_step', 0)
        curr_idx = min(curr_ptr, len(df) - 1)

        history_records.append({
            'step': step_idx,
            'price': df.iloc[curr_idx]['close'],
            'position': info.get('position', 0),
            'net_worth': info.get('portfolio_valuation', 1.0),
            'date': df.index[curr_idx],
        })
        if done[0]: break

    # --- 绘图界面 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(bottom=0.18, hspace=0.3)

    # 背景参考线
    all_prices = [h['price'] for h in history_records]
    all_nws = [h['net_worth'] for h in history_records]

    ax1.plot(all_prices, color='gray', alpha=0.3, label='Price History', zorder=1)
    ax2.plot(all_nws, color='#2ca02c', label='Portfolio Value', linewidth=1.5)

    # 动态标记点
    curr_marker, = ax1.plot([], [], 'ko', ms=6, zorder=15)
    dynamic_elements = []

    # 滑动条控制
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Time Step', 0, len(history_records) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        h = history_records[idx]

        # 清除上一帧的箭头和文本
        for el in dynamic_elements:
            el.remove()
        dynamic_elements.clear()

        # 更新当前价位点
        curr_marker.set_data([idx], [h['price']])

        # 绘制动作标记
        pos = h['position']
        if pos != 0:
            color = '#e31a1c' if pos < 0 else '#1f78b4'  # 红空蓝多
            marker = 'v' if pos < 0 else '^'
            # 箭头稍微偏移价格显示
            offset = h['price'] * 0.01 * (1 if pos > 0 else -1)

            # 散点标记箭头
            sc = ax1.scatter(idx, h['price'] + offset, marker=marker, s=150,
                             c=color, edgecolors='white', linewidths=1, zorder=20)

            # 悬浮文字
            txt = ax1.text(idx + 2, h['price'], f"POS: {pos}\nNW: {h['net_worth']:.3f}",
                           color='white', fontsize=10, fontweight='bold',
                           bbox=dict(facecolor=color, alpha=0.8, edgecolor='none'))

            dynamic_elements.extend([sc, txt])

        ax1.set_title(f"Visualizer | Date: {h['date']} | Net Worth: {h['net_worth']:.4f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    ax1.set_ylabel("Asset Price")
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Net Worth")
    ax2.grid(True, alpha=0.3)

    print(">>> 可视化窗口已就绪")
    plt.show()


if __name__ == "__main__":
    # 请确保文件名与你训练保存的一致
    interactive_visualize(
        model_path="eth_ppo_v4_sharpe.zip",
        vec_path="vec_eth_ppo_v4_sharpe.pkl",
        train_data_path="processed_train_data.csv"
    )