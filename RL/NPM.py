import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import json
from calculate_max_potential import calculate_max_potential_sharpe
from torch.utils.tensorboard import SummaryWriter

# ====================== Positional Encoding ======================
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
        return x + self.pe[:, :x.size(1)]


# ====================== Transformer Decoder Backbone ======================
class TransformerDecoderBackbone(nn.Module):
    def __init__(self, input_dim=29, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=60)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        output = self.transformer_decoder(tgt=x, memory=x, tgt_mask=tgt_mask)
        output = self.norm(output)
        return output[:, -1, :]


# ====================== 主模型 ======================
class NestedPPOModel(nn.Module):
    def __init__(self, d_model=256, detail_dim=100, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.backbone = TransformerDecoderBackbone(
            input_dim=29, d_model=d_model, nhead=8, num_layers=4,
            dim_feedforward=512, dropout=0.1
        )

        # L1: Action Head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3)
        )

        # L2: State Head - 增加深度和 LayerNorm，强制提取逻辑特征
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  # 稳定来自 Backbone 的特征分布
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 7)
        )

        # L3: Detail Head - 深度与 L2 保持一致，处理更细碎的分类逻辑
        self.detail_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, detail_dim)
        )

        # Value Head - 评价网络，通常需要一定深度来拟合净值曲线
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], self.seq_len, -1)
        features = self.backbone(x)

        return (self.action_head(features),
                self.state_head(features),
                self.detail_head(features),
                self.value_head(features))

# ====================== Agent ======================
class Agent(nn.Module):
    def __init__(self, lr=5e-5, gamma=0.99, K_epochs=10, eps_clip=0.2, entropy_coef=0.01):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.w_action = 1.0
        self.w_value = 0.5
        self.w_state = 0.0
        self.w_detail = 0.0
        self.policy = NestedPPOModel()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = NestedPPOModel()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update_weights(self,
                                   min_combo_rate: float,
                                   has_annotations: bool = False,
                                   l2_correct: int = 0,
                                   l2_total: int = 0,
                                   current_avg_dsr: float = 0):
        # ==================== 第一层：DSR 总开关 ====================
        if current_avg_dsr <= 0:
            self.w_state = 0.0
            self.w_detail = 0.0
            l2_acc = (l2_correct / l2_total) if l2_total > 0 else 0.0
            print(f" [挂起] DSR={current_avg_dsr:.4f} (<=0)，优先优化盈利能力。L2权重已抑制。")

        else:
            # ==================== 第二层：L2 动态调整 (DSR > 0 时) ====================
            if l2_total == 0:
                self.w_state = 0.2  # 初始启动权重
                l2_acc = 0.0
            else:
                l2_acc = l2_correct / l2_total

                if l2_correct == l2_total:  # L2 完美
                    self.w_state = 0.02
                elif l2_acc >= 0.98:  # 接近完美
                    self.w_state = 0.08
                elif l2_acc >= 0.95:
                    self.w_state = 0.15
                elif min_combo_rate < 0.85:  # 组合表现差，强行拉升
                    self.w_state = 3.0
                elif min_combo_rate < 0.92:
                    self.w_state = 1.5
                else:
                    self.w_state = 0.30  # 正常监督

            # ==================== 第三层：L3 细节逻辑 ====================
            # 只有当 L2 基础打好了（min_combo_rate >= 0.95），才开启 L3
            if has_annotations and min_combo_rate >= 0.95:
                self.w_detail = 0.4
            else:
                self.w_detail = 0.02

        # ====================== 打印信息 ======================
        if l2_total > 0:
            print(f" → Loss Weights: L1={self.w_action:.2f} | L2={self.w_state:.2f} | "
                  f"L3={self.w_detail:.2f} | MinCombo={min_combo_rate:.1%} | "
                  f"Overall L2={l2_correct}/{l2_total} ({l2_acc:.1%})")
        else:
            print(f" → Loss Weights: L1={self.w_action:.2f} | L2={self.w_state:.2f} | "
                  f"L3={self.w_detail:.2f} | MinCombo={min_combo_rate:.1%} | L2=0/0")

    def get_masked_logits(self, logits, mask):
        masked = logits.clone()
        masked[~mask] = -1e8
        return masked

    def get_state_mask(self, current_pos_normed):
        batch_size = current_pos_normed.size(0)
        device = current_pos_normed.device
        return torch.ones((batch_size, 7), dtype=torch.bool, device=device)


# ====================== 交易环境 ======================
class TradingEnv:
    def __init__(self, data, initial_balance=10000, fee=0.0005, history_len=60):
        # 原始数据加载
        self.data = data
        self.prices = self.data[:, 3]
        self.history_len = history_len
        self.initial_balance = initial_balance
        self.fee = fee
        self.n_steps = len(self.prices)

        # 微分夏普比率相关变量
        self.running_mean = 0.0
        self.running_var = 0.0
        self.eta = 0.02

        # --- 局部标准化所需的在线统计变量 ---
        self.feature_dim = self.data.shape[1]
        self.step_count = 0
        self.online_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.online_var = np.zeros(self.feature_dim, dtype=np.float32)
        # ----------------------------------

        self.reset()

    def reset(self):
        self.current_step = 0
        self.pos = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.history = []

        # 重置在线统计量（也可选择不重置以保留跨episode的统计特性，但在RL中通常重置）
        self.step_count = 0
        self.online_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.online_var = np.zeros(self.feature_dim, dtype=np.float32)

        return self._get_obs()

    def _get_obs(self):
        # 1. 获取当前原始特征
        raw_feat = self.data[self.current_step]

        # 2. 更新局部统计量 (Welford's Online Algorithm)
        self.step_count += 1
        last_mean = self.online_mean.copy()
        self.online_mean += (raw_feat - last_mean) / self.step_count
        self.online_var += (raw_feat - last_mean) * (raw_feat - self.online_mean)

        # 3. 计算标准差并标准化当前特征
        std = np.sqrt(self.online_var / self.step_count) + 1e-8
        current_feat = (raw_feat - self.online_mean) / std

        # 4. 构建交易状态信号
        pos_signal = {0: 0.0, 1: 1.0, 2: -1.0}[self.pos]
        unrealized_pnl = 0.0
        if self.pos != 0:
            price = self.prices[self.current_step]
            side = 1 if self.pos == 1 else -1
            unrealized_pnl = side * (price - self.entry_price) / (self.entry_price + 1e-8)

        # 5. 合并特征
        current_obs = np.concatenate([current_feat, [pos_signal], [unrealized_pnl]])

        # 6. 更新历史窗口
        self.history.append(current_obs)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        # 7. 填充 Padding (27个特征 + 1个仓位 + 1个盈亏 = 29维)
        padded = np.zeros((self.history_len, 29), dtype=np.float32)
        start = self.history_len - len(self.history)
        padded[start:] = np.array(self.history)

        return padded.flatten()

    def get_action_mask(self):
        mask = np.array([True, True, True], dtype=bool)
        if self.pos == 1:  # 持有多单，禁止开空
            mask[2] = False
        if self.pos == 2:  # 持有空单，禁止开多
            mask[1] = False
        return mask

    def get_state_label(self, action: int, current_pos: int) -> int:
        if current_pos == 0:
            if action == 0: return 0
            if action == 1: return 1
            if action == 2: return 2
        elif current_pos == 1:
            if action == 1: return 3
            if action == 0: return 5
            return 3
        elif current_pos == 2:
            if action == 2: return 4
            if action == 0: return 6
            return 4
        return 0

    def step(self, action):
        prev_net_worth = self.net_worth
        self.current_step += 1

        done = self.current_step >= self.n_steps - 1
        if done:
            return self._get_obs(), 0.0, True

        price = self.prices[self.current_step]

        # 交易逻辑处理
        if self.pos == 0:
            if action == 1:
                self.pos = 1
                self.entry_price = price
                self.balance -= self.balance * self.fee
            elif action == 2:
                self.pos = 2
                self.entry_price = price
                self.balance -= self.balance * self.fee
        elif self.pos == 1:
            if action == 0:
                self.balance *= (1 + (price - self.entry_price) / self.entry_price - self.fee)
                self.pos = 0
        elif self.pos == 2:
            if action == 0:
                self.balance *= (1 + (self.entry_price - price) / self.entry_price - self.fee)
                self.pos = 0

        # 计算当前净值
        if self.pos == 1:
            self.net_worth = self.balance * (1 + (price - self.entry_price) / self.entry_price)
        elif self.pos == 2:
            self.net_worth = self.balance * (1 + (self.entry_price - price) / self.entry_price)
        else:
            self.net_worth = self.balance

        # 计算奖励 (微分夏普)
        R_t = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        reward = self._calculate_differential_sharpe(R_t)

        return self._get_obs(), reward, done

    def _calculate_differential_sharpe(self, R_t):
        delta = R_t - self.running_mean
        self.running_mean += self.eta * delta
        self.running_var += self.eta * (R_t ** 2 - self.running_var)
        sigma = np.sqrt(max(self.running_var - self.running_mean ** 2, 1e-8))
        reward = (self.running_var * delta - 0.5 * self.running_mean * (R_t ** 2 - self.running_var)) / (
                    sigma ** 3 + 1e-8)
        return np.clip(reward, -1.0, 1.0)

# ====================== Gym 包装 ======================
class TradingGymEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.inner = TradingEnv(data)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(1740,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, **kwargs):
        obs = self.inner.reset()
        return obs, {
            "action_mask": self.inner.get_action_mask().astype(np.float32),
            "current_step": self.inner.current_step
        }

    def step(self, action):
        prev_pos_raw = self.inner.pos
        state_label = self.inner.get_state_label(action, prev_pos_raw)
        obs, reward, done = self.inner.step(action)
        info = {
            "state_label": state_label,
            "prev_pos": prev_pos_raw,
            "current_pos": self.inner.pos,
            "portfolio_valuation": self.inner.net_worth,
            "action_mask": self.inner.get_action_mask().astype(np.float32),
            "current_step": self.inner.current_step,
            "current_price": self.inner.prices[self.inner.current_step]  # <-- 新增：直接把价格传出来
        }
        return obs, reward, done, False, info

def evaluate_policy(agent, eval_data, device, manual_l3_labels, split_offset, show=0):
    """
    使用 eval_data 进行考试，生成详细的 L2 审计矩阵和 L3 准确率
    新增：show=1 时，在评估**完成后**保存一次完整详细 JSON 日志
    """
    # 局部创建评估环境
    eval_env = TradingGymEnv(eval_data)
    agent.policy.eval()
    obs, info = eval_env.reset()
    done = False

    # 1. 初始化统计容器 [动作(3), 预测类别(7)]
    l2_audit_total = np.zeros((3, 7))
    l2_audit_success = np.zeros((3, 7))
    l3_correct = 0
    l3_count = 0
    steps = 0

    # 安全获取初始仓位
    p_pos = info.get("current_pos", 0)

    # 合理的 (仓位, 动作) -> L2 映射
    reasonable_map = {
        (0, 0): 0, (0, 1): 1, (0, 2): 2,
        (1, 1): 3, (1, 0): 5,
        (2, 2): 4, (2, 0): 6,
    }

    # ==================== 新增：评估详细日志收集 ====================
    eval_log = [] if show else None

    with torch.no_grad():
        while not done:
            # 获取动作掩码
            mask_val = info.get("action_mask", np.ones(3))
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            mask = torch.as_tensor(mask_val, dtype=torch.bool).unsqueeze(0).to(device)

            # 推理
            action_logits, state_logits, detail_logits, _ = agent.policy(obs_t)
            masked_logits = agent.get_masked_logits(action_logits, mask)
            action = torch.argmax(masked_logits, dim=-1).item()
            pred_l2 = torch.argmax(state_logits, dim=-1).item()
            pred_l3 = torch.argmax(detail_logits, dim=-1).item()

            # 环境步进
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            c_pos = info.get("current_pos", 0)

            # --- 核心审计逻辑 ---
            is_reasonable = False
            if (p_pos, action) in reasonable_map and pred_l2 == reasonable_map[(p_pos, action)]:
                is_reasonable = True
            if pred_l2 in (5, 6) and c_pos != 0:
                is_reasonable = False

            # 更新审计矩阵
            l2_audit_total[action, pred_l2] += 1
            if is_reasonable:
                l2_audit_success[action, pred_l2] += 1

            # --- L3 标注统计 ---
            current_env_step = info.get("current_step", 0)
            global_step = current_env_step + split_offset
            if global_step in manual_l3_labels:
                l3_count += 1
                if pred_l3 == manual_l3_labels[global_step]:
                    l3_correct += 1

            # ==================== 收集详细日志（仅 show=1 时） ====================
            if show:
                eval_log.append({
                    "global_step": global_step,
                    "local_step": current_env_step,
                    "price": float(info.get("current_price", 0.0)),
                    "action": action,
                    "pos_before": p_pos,
                    "pos_after": c_pos,
                    "net_worth": float(info.get("portfolio_valuation", 10000.0)),
                    "pred_l2": pred_l2,
                    "true_l2": int(info.get("state_label", pred_l2)),
                    "pred_l3": pred_l3,
                    "is_reasonable": is_reasonable,
                    "reward": float(reward)
                })

            steps += 1
            p_pos = c_pos

    # ==================== 评估完成后保存一次 JSON ====================
    if show and eval_log:
        log_path = "evaluate_mario_state.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(eval_log, f, ensure_ascii=False, indent=2)
        print(f"📊 评估完成！完整详细日志已保存 → {log_path} （共 {len(eval_log)} 条记录）")

    # 最终净值
    final_net_worth = info.get("portfolio_valuation", 10000.0)
    agent.policy.train()

    return {
        "net_worth": float(final_net_worth),
        "l2_audit_total": l2_audit_total,
        "l2_audit_success": l2_audit_success,
        "l2_audit_success_count": int(l2_audit_success.sum()),
        "steps": steps,
        "l3_acc": l3_correct / l3_count if l3_count > 0 else 0.0,
        "l3_count": l3_count
    }
def extract_from_infos(infos, key, num_envs, default_value=0, dtype=torch.long, device="cpu"):
    if isinstance(infos, dict) and key in infos:
        return torch.as_tensor(infos[key], dtype=dtype, device=device)
    if isinstance(infos, (list, tuple)):
        vals = [i.get(key, default_value) for i in infos]
        return torch.tensor(vals, dtype=dtype, device=device)
    return torch.full((num_envs,), default_value, dtype=dtype, device=device)


def load_l3_annotations(model_path):
    """专门负责加载和解析 L3 标注数据"""
    annotation_json = f"{model_path}_l3_annotation_log_2.json"
    manual_labels = {}

    if os.path.exists(annotation_json):
        with open(annotation_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 提取有评论的标注
        manual_labels = {
            item["step"]: item["detail_l3"]
            for item in data if item.get("comment", "").strip() != ""
        }

    has_annotations = len(manual_labels) > 0
    return manual_labels, has_annotations
# ====================== 主训练函数 ======================
def run(TRAIN_DATA_PATH="ppo_x_v1.npy",
        MODEL_PATH="ppo_v1",
        num_envs=8,
        total_timesteps=50_000_000,
        n_steps=5124,
        batch_size=2048,
        n_epochs=10,
        patience_limit=20,
        show = 0):
    manual_l3_labels, has_l3_annotations = load_l3_annotations(MODEL_PATH)
    if has_l3_annotations:
        print(f"✅ 发现 {len(manual_l3_labels)} 条人工标注，进入 L3 专项强化模式！")
    else:
        print("⚠️ 未找到有效标注，L3 将保持无监督。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l2_weights = torch.tensor([1.0, 5.0, 5.0, 0.2, 0.2, 5.0, 5.0], device=device)
    print(f"Using device: {device}")

    full_data = np.load(TRAIN_DATA_PATH).astype(np.float32)
    split_idx = int(len(full_data) * 0.8)
    train_data = full_data[:split_idx]
    eval_data = full_data[split_idx:]
    envs = gym.vector.AsyncVectorEnv([lambda: TradingGymEnv(train_data) for _ in range(num_envs)])
    agent = Agent().to(device)

    potential = calculate_max_potential_sharpe(TRAIN_DATA_PATH)
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=5e-5, eps=1e-5)
    writer = SummaryWriter(f"runs/{MODEL_PATH}")
    checkpoint_file = f"{MODEL_PATH}.pt"
    start_update = 0
    best_net_worth = -float('inf')
    patience_counter = 0
    l2_audit_success = np.zeros((3, 7))
    l2_audit_total = np.zeros((3, 7))

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        agent.policy.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'policy_old_state_dict' in checkpoint:
                agent.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
            print(f"从 update {start_update} 继续训练...")
        else:
            print("checkpoint缺少optimizer状态，从头开始训练...")

    # Buffers
    obs_shape = (1740,)
    obs = torch.zeros((n_steps, num_envs) + obs_shape, device=device)
    actions = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((n_steps, num_envs), device=device)
    rewards = torch.zeros((n_steps, num_envs), device=device)
    dones = torch.zeros((n_steps, num_envs), device=device)
    values = torch.zeros((n_steps, num_envs), device=device)
    masks = torch.zeros((n_steps, num_envs, 3), dtype=torch.bool, device=device)
    state_labels = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)
    data_steps = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)

    next_obs, next_info = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)
    if isinstance(next_info, dict) and "current_step" in next_info:
        next_step_idx = torch.as_tensor(next_info["current_step"], dtype=torch.long, device=device)
    else:
        next_step_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    print("=== Transformer Decoder PPO 训练开始 ===")

    for update in range(start_update + 1, total_timesteps // (n_steps * num_envs) + 1):
        l2_audit_total.fill(0)
        l2_audit_success.fill(0)
        for step in range(n_steps):
            obs[step] = next_obs
            dones[step] = next_done
            data_steps[step] = next_step_idx
            if isinstance(next_info, dict) and "action_mask" in next_info:
                masks[step] = torch.as_tensor(next_info["action_mask"], dtype=torch.bool, device=device)
            else:
                masks[step] = torch.ones((num_envs, 3), dtype=torch.bool, device=device)

            with torch.no_grad():
                action_logits, state_logits, detail_logits, value = agent.policy(next_obs)
                masked_logits = agent.get_masked_logits(action_logits, masks[step])
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = value.squeeze(-1)
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
            state_labels[step] = extract_from_infos(infos, "state_label", num_envs, device=device)
            next_step_idx = extract_from_infos(infos, "current_step", num_envs, device=device)

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(term_np | trunc_np, dtype=torch.float32, device=device)
            rewards[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            masks[step] = extract_from_infos(infos, "action_mask", num_envs,
                                             default_value=[True, True, True],
                                             dtype=torch.bool, device=device)
            next_info = infos

            if step % 200 == 0:
                v_array = infos.get("portfolio_valuation", np.full(num_envs, 10000.0)) if isinstance(infos,
                                                                                                     dict) else np.full(
                    num_envs, 10000.0)
                avg_net = float(np.mean(v_array))
                print(f"Update {update:4d} |step {step}| Avg Net Worth: {avg_net:10.2f}")

        # GAE
        with torch.no_grad():
            _, _, _, next_value = agent.policy(next_obs)
            next_value = next_value.squeeze(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + agent.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + agent.gamma * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values

        # PPO Update
        b_obs = obs.reshape((-1,) + obs_shape)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_masks = masks.reshape(-1, 3)
        b_state_labels = state_labels.reshape(-1)
        b_data_steps = data_steps.reshape(-1)
        b_inds = np.arange(b_obs.shape[0])

        # ====================== PPO 优化循环 ======================
        agent.policy_old.load_state_dict(agent.policy.state_dict())
        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, b_obs.shape[0], batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]
                action_logits, state_logits, detail_logits, newvalue = agent.policy(b_obs[mb_inds])
                masked_logits = agent.get_masked_logits(action_logits, b_masks[mb_inds])
                dist = Categorical(logits=masked_logits)
                newlogprob = dist.log_prob(b_actions[mb_inds])
                entropy = dist.entropy().mean()
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = (b_advantages[mb_inds] - b_advantages[mb_inds].mean()) / (
                            b_advantages[mb_inds].std() + 1e-8)
                pg_loss = torch.max(-mb_advantages * ratio,
                                    -mb_advantages * torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip)).mean()
                v_loss = 0.5 * ((newvalue.squeeze(-1) - b_returns[mb_inds]) ** 2).mean()

                # L2 部分
                loss_state = F.cross_entropy(
                    state_logits,
                    b_state_labels[mb_inds],
                    weight=l2_weights,
                    reduction='mean'
                )

                # L3 部分
                mb_data_steps_np = b_data_steps[mb_inds].cpu().numpy()
                target_l3_list = []
                valid_mask = []
                for s_idx in mb_data_steps_np:
                    if s_idx in manual_l3_labels:
                        target_l3_list.append(manual_l3_labels[s_idx])
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
                detail_dist = Categorical(logits=detail_logits)
                loss_detail_entropy = 0.01 * detail_dist.entropy().mean()
                if valid_mask.any():
                    target_l3_tensor = torch.tensor(target_l3_list, dtype=torch.long, device=device)
                    loss_l3_supervised = F.cross_entropy(detail_logits[valid_mask], target_l3_tensor)
                    loss_detail = loss_detail_entropy + 20.0 * loss_l3_supervised
                else:
                    loss_detail = loss_detail_entropy

                loss = (agent.w_action * pg_loss +
                        agent.w_value * v_loss +
                        agent.w_state * loss_state +
                        agent.w_detail * loss_detail -
                        agent.entropy_coef * entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
                optimizer.step()


        # ====================== 早停判断 ======================
        # ====================== 早停与评估判断 (修改后) ======================
        if update % 1 == 0:
            # 1. 在测试集上跑全量评估
            eval_results = evaluate_policy(
                agent,
                eval_data,
                device,
                manual_l3_labels=manual_l3_labels,
                split_offset=split_idx,
                show = show
            )

            # 2. 提取评估指标 (注意：确保 evaluate_policy 返回了这些 key)
            current_net_worth = eval_results["net_worth"]
            eval_l2_correct = eval_results["l2_audit_success_count"]  # 建议用 count 区分
            eval_l2_total = eval_results["steps"]

            # 获取详细的审计矩阵用于打印分类报告
            l2_audit_total = eval_results["l2_audit_total"]
            l2_audit_success = eval_results["l2_audit_success"]

            # ==================== L2 详细统计 (审计矩阵解析) ====================
            active_mask = l2_audit_total > 0
            # 避免除以 0
            combo_rates = np.divide(l2_audit_success, l2_audit_total,
                                    out=np.zeros_like(l2_audit_success, dtype=float),
                                    where=active_mask)

            min_combo_rate = np.min(combo_rates[active_mask]) if np.any(active_mask) else 0.0
            avg_sanity_rate = np.mean(combo_rates[active_mask]) if np.any(active_mask) else 0.0

            # 计算 7 个分类的汇总准确率
            l2_class_total = l2_audit_total.sum(axis=0)
            l2_class_correct = l2_audit_success.sum(axis=0)

            # 打印分类报告
            print("\n" + "=" * 30)
            print("L2 Audit Report per class:")
            for cls in range(7):
                if l2_class_total[cls] > 0:
                    acc = l2_class_correct[cls] / l2_class_total[cls]
                    print(f"  {cls}: {int(l2_class_correct[cls])}/{int(l2_class_total[cls])} ({acc:.1%})", end="    ")
                else:
                    print(f"  {cls}: 0/0 (---%)", end="    ")
                if (cls + 1) % 3 == 0: print()
            print(f"\nMinCombo: {min_combo_rate:.1%} | Overall L2: {eval_l2_correct}/{eval_l2_total}")

            # ==================== 动态 Class Weights (针对训练 Batch) ====================
            # 注意：b_state_labels 是本轮训练的最后一部分样本，用于平衡 L2 的梯度
            with torch.no_grad():
                unique, counts = torch.unique(b_state_labels, return_counts=True)
                cw = torch.ones(7, device=device) * 0.1
                total_samples = b_state_labels.size(0)
                for u, c in zip(unique, counts):
                    cw[u.item()] = total_samples / (7.0 * c.float() + 1e-8)
                l2_weights = cw / (cw.mean() + 1e-8)
                l2_weights = torch.clamp(l2_weights, min=0.01, max=20.0)

            # ==================== 动态权重调整与模型保存 ====================
            current_avg_dsr = float(torch.mean(rewards)) - potential * 0.1

            agent.update_weights(
                min_combo_rate=min_combo_rate,
                has_annotations=has_l3_annotations,
                l2_correct=eval_l2_correct,
                l2_total=eval_l2_total,
                current_avg_dsr=current_avg_dsr
            )

            print(f"Update {update} | L2 Weight: {agent.w_state:.2f} | L3 Weight: {agent.w_detail:.2f}")

            # 4. 判断是否改进：必须是【测试集】表现更好才保存
            # 进阶建议：不仅看净值，还可以要求 eval_l2_acc > 0.5 才保存，防止瞎撞
            improved = current_net_worth > best_net_worth

            if improved:
                best_net_worth = current_net_worth
                patience_counter = 0

                # 保存模型
                torch.save({
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'policy_old_state_dict': agent.policy_old.state_dict(),
                    'update': update,
                    'best_net_worth': best_net_worth,
                    'w_state': agent.w_state,
                    'w_detail': agent.w_detail
                }, checkpoint_file)
                print(f"💎 发现更好的泛化模型，测试集净值: {best_net_worth:.2f}，模型已保存")
            else:
                patience_counter += 1
                print(f"⏳ 测试集未提升，耐心值: {patience_counter}/{patience_limit}")

            if patience_counter >= patience_limit:
                print(f"🛑 [早停] 测试集表现长期未提升，停止训练以防止过拟合。")
                break

    envs.close()
    writer.close()
    print("训练完成！")

    # ====================== 生成 L3 标注日志 ======================
    print("\n训练完成！正在生成 L3 标注日志...")
    test_env = TradingGymEnv(train_data)
    obs, _ = test_env.reset()
    done = False
    step_idx = 0
    annotation_log = []
    with torch.no_grad():
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action_logits, state_logits, detail_logits, _ = agent.policy(obs_t)
            l1 = torch.argmax(action_logits).item()
            l2 = torch.argmax(state_logits).item()
            l3 = torch.argmax(detail_logits).item()
            obs, reward, done, _, info = test_env.step(l1)
            annotation_log.append({
                "step": step_idx,
                "pos": test_env.inner.pos,
                "action_l1": l1,
                "state_l2": l2,
                "detail_l3": l3,
                "reward": float(reward),
                "net_worth": float(info["portfolio_valuation"]),
                "comment": ""
            })
            step_idx += 1

    with open(f"{MODEL_PATH}_l3_annotation_log_2.json", "w", encoding="utf-8") as f:
        json.dump(annotation_log, f, ensure_ascii=False, indent=2)
    print(f"L3 标注日志已生成：{f"{MODEL_PATH}_l3_annotation_log_2.json"}")


if __name__ == "__main__":
    run(show=1)