import os
import torch
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

# 导入你原有的包装器
from PPO import FeatureWrapper, MultiLevelHardStopWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def continuous_train(# 配置路径
        OLD_MODEL_PATH = "eth_ppo_end.zip",
        OLD_VEC_PATH = "vec_eth_ppo_end.pkl",
        FINETUNE_DATA_PATH = "processed_train_data.csv",
        MODEL_PATH = "ppo_trading_model_v3",
        VEC_PATH = "vec_normalize_v3.pkl"
    ):
    # 1. 加载数据并提取特征列
    df = pd.read_csv(FINETUNE_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    feature_columns = [f"f_{i}" for i in range(9)]

    # 2. 环境重建
    def make_env():
        base_env = gym.make('TradingEnv', df=df, trading_fees=0.0005, windows=None)
        env = FeatureWrapper(base_env, feature_columns)
        env = MultiLevelHardStopWrapper(env)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([make_env])

    # 加载 VecNormalize 并保持训练模式
    if os.path.exists(OLD_VEC_PATH):
        venv = VecNormalize.load(OLD_VEC_PATH, venv)
        venv.training = True
    else:
        print("警告: 缺少标准化参数文件，这可能导致观测数据尺度不一致。")

    # 3. 加载模型 (不重置学习率)
    if not os.path.exists(OLD_MODEL_PATH):
        return

    # 直接加载，不传入 custom_objects 里的学习率修改，则沿用保存时的参数
        # 注入微调参数：低学习率 + 高探索度
    custom_objects = {
            "ent_coef": 0.02,
            "clip_range": 0.1
        }

    model = PPO.load(OLD_MODEL_PATH, env=venv, device=DEVICE, custom_objects=custom_objects)

    # 4. 回调函数 (维持你的早停逻辑)
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=50,  # 维持原代码设置
        min_evals=30,
        verbose=1,
    )

    eval_callback = EvalCallback(
        venv,
        best_model_save_path="./logs/finetune_best",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=stop_train_callback
    )

    print("开始持续训练...")
    model.learn(
        total_timesteps=5000000,
        callback=eval_callback,
        reset_num_timesteps=False  # 关键：在 Tensorboard 中衔接曲线
    )

    model.save(MODEL_PATH)
    venv.save(VEC_PATH)
    print("持续训练完成！")


def run(# 配置路径
        OLD_MODEL_PATH = "eth_ppo_v4.zip",
        OLD_VEC_PATH = "vec_eth_ppo_v4.pkl",
        FINETUNE_DATA_PATH = "processed_train_data.csv",
        MODEL_PATH = "eth_ppo_v5.zip",
        VEC_PATH = "vec_eth_ppo_v5.pkl"
    ):
    continuous_train(OLD_MODEL_PATH,OLD_VEC_PATH,FINETUNE_DATA_PATH,MODEL_PATH,VEC_PATH)
if __name__ == "__main__":
    run()