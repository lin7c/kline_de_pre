import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mplfinance as mpf
import os
from scipy.fftpack import idct
from UDmodel import DiffusionUNet, GaussianDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_resources(GAF_FILE, DCT_FILE, X_RAW_FILE, DIFFUSION_MODEL_PATH, T_MODEL_PATH):
    print(">>> 正在加载资源...")
    X_gaf = np.load(GAF_FILE)
    Y_dct = np.load(DCT_FILE)
    X_raw = np.load(X_RAW_FILE)

    # 1. 加载 Transformer 模型
    from Dmodel import GafCnnTransformer
    t_model = GafCnnTransformer(output_dim=9).to(device)
    t_ckpt = torch.load(T_MODEL_PATH, map_location=device, weights_only=False)
    t_model.load_state_dict(t_ckpt.get('model_state_dict', t_ckpt))
    t_model.eval()

    # 2. 获取特征维度
    with torch.no_grad():
        temp_gaf = torch.from_numpy(X_gaf[0:1]).float().to(device)
        feat_sample = t_model.cnn(temp_gaf)
        feat_sample = nn.functional.adaptive_avg_pool2d(feat_sample, (1, 1)).view(1, -1)
        feat_dim = feat_sample.shape[1]

    # 3. 加载 Diffusion 模型
    model = DiffusionUNet(feature_dim=feat_dim).to(device)
    ckpt = torch.load(DIFFUSION_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()

    diffuser = GaussianDiffusion(timesteps=1000, device=device)

    return X_gaf, Y_dct, X_raw, t_model, model, diffuser


@torch.no_grad()
def p_sample_loop(model, diffuser, feats, dcts):
    shape = (feats.size(0), 60, 4)
    img = torch.randn(shape, device=device)

    print(">>> 正在进行扩散去噪采样...")
    for i in reversed(range(0, diffuser.timesteps)):
        t = torch.full((feats.size(0),), i, device=device, dtype=torch.long)
        pred_noise = model(img, t, dcts, feats)

        alpha = diffuser.alphas[i]
        alpha_bar = diffuser.alphas_cumprod[i]
        beta = diffuser.betas[i]

        if i > 0:
            noise = torch.randn_like(img)
        else:
            noise = 0

        coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        img = (1 / torch.sqrt(alpha)) * (img - coef * pred_noise) + torch.sqrt(beta) * noise

    return img.cpu().numpy().squeeze(0)


def run_inference(idx, X_gaf, Y_dct, X_raw, t_model, model, diffuser):
    gaf_tensor = torch.from_numpy(X_gaf[idx:idx+1]).float().to(device)

    # 提取特征
    with torch.no_grad():
        feat = t_model.cnn(gaf_tensor)
        feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).view(1, -1)

    # 使用原始 Y_dct（已局部标准化）
    dct_tensor = torch.from_numpy(Y_dct[idx:idx+1]).float().to(device)

    # 生成 Delta (Normalized)
    pred_delta_norm = p_sample_loop(model, diffuser, feat, dct_tensor)

    # ====================== 逆标准化（关键修改）======================
    local_std = np.std(X_raw[idx, :, 3]) + 1e-9

    # 重建趋势线（乘回 local_std）
    coeffs_1m = Y_dct[idx, :3]
    full_coeffs = np.zeros(60)
    full_coeffs[:3] = coeffs_1m
    trend_line = idct(full_coeffs, type=2, norm='ortho').reshape(60, 1) * local_std

    base_price = X_raw[idx, -1, 3]

    # Delta 逆标准化
    pred_delta = pred_delta_norm * local_std

    # 合成最终预测
    trend_ohlc = np.tile(trend_line, (1, 4))
    final_pred = base_price + trend_ohlc + pred_delta
    pure_trend = base_price + trend_ohlc

    true_future = X_raw[idx + 1: idx + 61, -1, :4]

    return true_future, final_pred, pure_trend

def plot_result(true_data, pred_data, trend_data):
    def to_df(data):
        df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
        df.index = pd.date_range(start='2026-04-09 09:00', periods=len(data), freq='min')
        return df

    df_true = to_df(true_data)
    df_pred = to_df(pred_data)

    mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--')

    fig = mpf.figure(style=s, figsize=(12, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # --- 关键修改：分别为 ax1 和 ax2 创建 addplot ---
    ap1 = mpf.make_addplot(trend_data[:, 3], color='blue', linestyle='--', width=1.5, ax=ax1)
    ap2 = mpf.make_addplot(trend_data[:, 3], color='blue', linestyle='--', width=1.5, ax=ax2)

    # 绘图时分别传入对应的 addplot
    mpf.plot(df_true, type='candle', ax=ax1, axtitle="Ground Truth", addplot=ap1)
    mpf.plot(df_pred, type='candle', ax=ax2, axtitle="Diffusion Generated (Skeleton + Delta)", addplot=ap2)

    print(">>> 还原完成。")
    mpf.show()

if __name__ == "__main__":
    config = {
        "GAF_FILE": "../RL/gaf_v1.npy",
        "DCT_FILE": "../CNN_Transformer/y_transformer_v1_g.npy",
        "X_RAW_FILE": "../CNN/input_x_v1.npy",
        "DIFFUSION_MODEL_PATH": "diffusion_delta_v1.pth",
        "T_MODEL_PATH": "../CNN_Transformer/transformer_dct_v1.pth"
    }

    X_gaf, Y_dct, X_raw, t_model, d_model, diffuser = load_all_resources(
        config["GAF_FILE"], config["DCT_FILE"], config["X_RAW_FILE"],
        config["DIFFUSION_MODEL_PATH"], config["T_MODEL_PATH"]
    )

    test_idx = np.random.randint(0, len(X_raw) - 61)
    print(f">>> 样本索引: {test_idx}")

    true_ohlc, pred_ohlc, trend_ohlc = run_inference(
        test_idx, X_gaf, Y_dct, X_raw, t_model, d_model, diffuser
    )

    plot_result(true_ohlc, pred_ohlc, trend_ohlc)