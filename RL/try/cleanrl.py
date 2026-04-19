import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import gymnasium as gym
import time

from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
# 假设这些是你原来的模块（请确保它们在同一目录或已正确导入）
from PPO import TradingDecoderExtractor, make_env


# ====================== CleanRL 风格的 Agent ======================
class Agent(nn.Module):
    def __init__(self, envs, df_for_stats=None,features_dim=256):
        super().__init__()
        # 1. 定义 buffer，这会让参数随 model.save() 一起保存
        # 默认初始化为 0 和 1
        self.register_buffer("means", torch.zeros(9))
        self.register_buffer("stds", torch.ones(9))
        self.register_buffer("reward_var", torch.ones(1))
        self.register_buffer("reward_count", torch.full((1,), 1e-8))
        self.ret = np.zeros(envs.num_envs)
        # 2. 如果传入了 df，则自动计算全局统计量（仅在第一次训练启动时）
        if df_for_stats is not None:
            feature_cols = [f"f_{i}" for i in range(9)]
            m = df_for_stats[feature_cols].mean().values
            s = df_for_stats[feature_cols].std().values
            self.means.copy_(torch.tensor(m, dtype=torch.float32))
            self.stds.copy_(torch.tensor(s, dtype=torch.float32))
            print(f"📊 已自动完成特征归一化初始化。")
        self.features_extractor = TradingDecoderExtractor(envs.single_observation_space, features_dim)

        # Actor: 输出 3 个席位，每个席位 3 种动作 → 总 logits 维度 9
        self.actor = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 3)
        )

        self.critic = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_features(self, x):
        # 自动化处理：无论 x 在 CPU 还是 GPU，means 都会自动随模型移动
        x_core = x[:,:, :9]
        x_others = x[:,:, 9:]
        # 执行归一化
        x_norm = (x_core - self.means) / (self.stds + 1e-8)
        if self.training:
            # 设置随机掩码概率 (例如 0.3 代表 30% 的时间让模型“失忆”)
            mask_prob = 0.3

            # 生成一个 (batch, 1, 1) 的随机张量
            # 结果为 0 的样本将丢失所有反馈信息，结果为 1 的样本保留
            mask = (torch.rand(x.shape[0], 1, 1, device=x.device) > mask_prob).float()

            # 应用掩码
            x_others = x_others * mask

            # 可选：给反馈特征增加极微小的高斯噪声 (0.001 级别)
            # 这样能让模型对“精确的 0”不那么敏感，增加实盘鲁棒性
            if torch.rand(1) > 0.8:
                x_others = x_others + torch.randn_like(x_others) * 1e-4
        # 拼接回原始维度 (如果是 9 维就不用拼接)
        x_combined = torch.cat([x_norm, x_others], dim=-1) if x.shape[-1] > 9 else x_norm
        return self.features_extractor(x_combined)

    def get_value(self, x):
        """用于 GAE 计算"""
        features = self.get_features(x)
        return self.critic(features).squeeze(-1)   # shape: (batch,)

    def get_action_and_value(self, x, action=None):
        # 确保 x 只搬运一次，并且 features 只提取一次
        features = self.get_features(x)

        # Actor 部分
        logits = self.actor(features)
        logits = logits.view(-1, 3, 3)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        logprob = probs.log_prob(action).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)

        # Critic 部分：直接复用 features
        value = self.critic(features).squeeze(-1)

        # weights 转换逻辑保持不变
        weights = action - 1

        return action, logprob, entropy, value, weights

# ====================== 训练主函数 ======================
def run(TRAIN_DATA_PATH="ppo_x_v1.csv", MODEL_PATH="eth_ppo_cleanrl_v1"):
    # 1. 加载数据
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ 找不到训练数据: {TRAIN_DATA_PATH}")
        return

    df = pd.read_csv(TRAIN_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    feature_columns = [f"f_{i}" for i in range(9)]

    # 2. 创建向量环境gf
    num_envs = 24
    envs = gym.vector.AsyncVectorEnv([
        lambda i=i: make_env(
            rank=i,
            seed=0,
            df_data=df,
            cols=feature_columns
        )() for i in range(num_envs)
    ])
    print("环境创建成功")
    # 3. 创建 Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = Agent(envs,df_for_stats=df, features_dim=256).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)

    # 4. TensorBoard
    writer = SummaryWriter(f"runs/{MODEL_PATH}")

    # 5. 超参数
    total_timesteps = 50_000_000
    n_steps = 2048
    batch_size = 4096          # 稍大一点，更稳定
    n_epochs = 10
    gamma = 0.97
    gae_lambda = 0.95
    ent_coef = 0.01           # 原来 0.05 偏高，建议降低
    clip_coef = 0.2
    vf_coef = 0.5
    max_grad_norm = 0.5
    best_net_worth = -float('inf')
    patience_limit = 30  # 30 个 Update 没新高就跑路
    patience_counter = 0
    checkpoint_file = f"{MODEL_PATH}.pt"

    if os.path.exists(checkpoint_file):
        print(f"📂 发现现有模型 {checkpoint_file}，正在加载并准备微调...")
        try:
            # map_location 确保在不同设备间切换（如 GPU 到 CPU）时不会报错
            checkpoint = torch.load(checkpoint_file, map_location=device,weights_only=False)

            # 加载模型权重
            agent.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态（这很重要，因为它保存了 Adam 的动量等信息，保证平滑微调）
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 获取之前的进度
            global_step = checkpoint.get('global_step', 0)
            # 计算新的起始 Update 轮次，避免 TensorBoard 时间轴重叠
            start_update = (global_step // (n_steps * num_envs)) + 1

            print(f"✅ 加载成功！从 Step {global_step} 继续训练。")
        except Exception as e:
            print(f"⚠️ 加载模型失败，将从零开始训练。错误原因: {e}")
    else:
        print("🆕 未发现预训练模型，开启全新训练任务。")
    # 6. 初始化 rollout 缓冲区
    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((n_steps, num_envs) + obs_shape).to(device)
    actions = torch.zeros((n_steps, num_envs, 3), dtype=torch.long).to(device)
    logprobs = torch.zeros((n_steps, num_envs)).to(device)
    rewards = torch.zeros((n_steps, num_envs)).to(device)
    dones = torch.zeros((n_steps, num_envs)).to(device)
    values = torch.zeros((n_steps, num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs, dtype=torch.float32).to(device)

    print("=== CleanRL + Transformer + TripleSlot PPO 开始训练 ===")
    print(f"Total timesteps: {total_timesteps} | Num envs: {num_envs} | Steps per rollout: {n_steps}")
    for update in range(1, total_timesteps // (n_steps * num_envs) + 1):
        epoch_final_values = []
        # === Rollout Phase ===
        for step in range(n_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            step_start = time.perf_counter()
            with torch.no_grad():
                action_idx, logprob, entropy, value, weights = agent.get_action_and_value(next_obs)

            values[step] = value
            actions[step] = action_idx
            logprobs[step] = logprob
            model_time = time.perf_counter() - step_start

            # --- 环境执行计时 ---
            env_start = time.perf_counter()
            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(weights.cpu().numpy())
            env_time = time.perf_counter() - env_start

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(term_np | trunc_np, dtype=torch.float32, device=device)

            # --- Reward Scaling ---
            agent.ret = agent.ret * gamma + reward_np

            # 简单的移动方差更新（如果你不想写复杂的 Welford）
            with torch.no_grad():
                # 计算本次 batch 的方差并更新到 buffer
                curr_var = torch.tensor(np.var(agent.ret), device=device)
                # 0.99 只是一个平滑系数，可以根据需要调整
                agent.reward_var = agent.reward_var * 0.99 + curr_var * 0.01

                # 缩放并存入 rewards
                std = torch.sqrt(agent.reward_var + 1e-8)
                rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device) / std

            # 重置 Episode 累计回报
            for i, d in enumerate(term_np | trunc_np):
                if d: agent.ret[i] = 0

            total_step_time = time.perf_counter() - step_start
            # 2. 在每一步都检查是否有环境“交卷”
            if any(term_np) or any(trunc_np):
                for i in range(num_envs):
                    if term_np[i] or trunc_np[i]:
                        # 既然 final_info 抓不到，我们直接抓当前这一步的实时净值
                        # 这在 TimeLimit 结束时和“最终净值”是几乎一样的
                        val = infos["portfolio_valuation"][i]

                        epoch_final_values.append(val)
                        print(f"✅ [强制捕获] 环境 {i} 结束 | 净值: {val:.4f} | 步数: {global_step}")
            if step % 200 == 0 or step < 10:
                avg_net_worth = np.mean(infos["portfolio_valuation"])
                avg_reward = float(np.mean(reward_np))  # 所有环境的平均收益
                max_reward = float(np.max(reward_np))  # 本次 step 中最好的环境收益
                print(f"Step {step:4d} | Model: {model_time * 1000:6.2f}ms | "
                      f"Env step: {env_time * 1000:6.2f}ms | Total: {total_step_time * 1000:6.2f}ms | "
                      f"Avg Reward: {avg_reward:+8.4f} | Max Reward: {max_reward:+8.4f} | Avg portfolio: {avg_net_worth:+8.4f}")

            # =============================================================

        # === 计算 GAE 与 Advantage ===
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0.0

            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # === Flatten ===
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 3))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # === PPO Update ===
        b_inds = np.arange(b_obs.shape[0])
        clipfracs = []

        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, b_obs.shape[0], batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                with torch.no_grad():
                    # 记录有多少比例的更新被 clip 了（这是 clipfrac 的定义）
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                    # 将当前的 clip 比例放入列表
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # --- 关键：从 infos 中提取本轮所有环境的平均最终净值 ---
        current_step_net_worth = np.mean(infos["portfolio_valuation"])
        if len(epoch_final_values) > 0:
            # 这里的 avg_net_worth 不再是“结算成绩”，而是“当前快照”
            avg_net_worth = current_step_net_worth

            # 逻辑：发现更高的赚钱参数
            if avg_net_worth > best_net_worth:
                best_net_worth = avg_net_worth
                patience_counter = 0  # 重置耐心

                # 保存模型
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'net_worth': best_net_worth,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }, checkpoint_file)
                print(f"💰 [新纪录] Update {update} | 当前净值: {best_net_worth:.4f} | 模型已保存")
            else:
                patience_counter += 1
                # 降低打印频率，避免刷屏
                if update % 10 == 0:
                    print(
                        f"⏳ 净值({avg_net_worth:.4f}) 未破纪录({best_net_worth:.4f}) | 耐心: {patience_counter}/{patience_limit}")

        # --- 触发早停 ---
        if patience_counter >= patience_limit:
            print(f"🛑 [监控早停] 连续 {patience_limit} 次评估未见净值增长。停止训练以防过拟合或崩溃。")
            break
        # === Logging ===
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if update:
            print(f"Update {update:4d} | Step {global_step:8d} | "
                  f"Value Loss: {v_loss.item():.4f} | Policy Loss: {pg_loss.item():.4f} | "
                  f"Entropy: {entropy_loss.item():.4f} | Explained Var: {explained_var:.3f} | Current net worth: {current_step_net_worth:.4f}")

    # === 保存模型 ===
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, f"{MODEL_PATH}.pt")

    print(f"%n>>> 训练完成！模型已保存至: {MODEL_PATH}.pt")
    envs.close()
    writer.close()


if __name__ == "__main__":
    run(TRAIN_DATA_PATH="processed_train_data.csv",MODEL_PATH="eth_ppo_cleanrl_v2")