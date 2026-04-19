import os
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from TPmodel import GafRegressionCNN
from Dmodel import GafCnnTransformer
from pyts.image import GramianAngularField


def precompute_gaf(input_npy_path, output_gaf_path, window_size=60):
    raw_data = np.load(input_npy_path).astype(np.float32)
    num_samples, _, num_channels = raw_data.shape
    gaf_all = np.empty((num_samples, num_channels, window_size, window_size), dtype=np.float32)
    gaf_tool = GramianAngularField(image_size=window_size, method='summation', sample_range=(-1, 1))

    for i in tqdm(range(num_samples), desc="计算 GAF (NCHW)"):
        window = raw_data[i]
        for ch in range(num_channels):
            series = window[:, ch].reshape(1, -1)
            s_min, s_max = series.min(), series.max()
            series_norm = (series - s_min) / (s_max - s_min) * 2 - 1 if (s_max - s_min) > 1e-9 else np.zeros_like(
                series)
            gaf_all[i, ch, :, :] = gaf_tool.fit_transform(series_norm)[0]

    np.save(output_gaf_path, gaf_all)
    print(f"GAF 预计算完成: {gaf_all.shape}")


def get_model_features(npy_path, gaf_npy_path, csv_path, trend_weights, reg_weights, device):
    gaf_data = np.load(gaf_npy_path).astype(np.float32)
    full_df = pd.read_csv(csv_path)

    # 加载模型
    t_net = GafCnnTransformer(input_channels=12, output_dim=9).to(device).eval()
    r_net = GafRegressionCNN(input_channels=12, output_dim=6).to(device).eval()

    t_ckpt = torch.load(trend_weights, map_location=device, weights_only=False)
    t_net.load_state_dict(t_ckpt.get('model_state_dict', t_ckpt))
    r_ckpt = torch.load(reg_weights, map_location=device, weights_only=False)
    r_net.load_state_dict(r_ckpt.get('model_state_dict', r_ckpt))

    # 特征推理
    all_features = []
    with torch.no_grad():
        for i in tqdm(range(len(gaf_data)), desc="特征推理"):
            gt = torch.from_numpy(gaf_data[i]).unsqueeze(0).to(device)
            t_out = t_net(gt).cpu().numpy().flatten()
            r_out = r_net(gt).cpu().numpy().flatten()
            all_features.append(np.concatenate([t_out, r_out]))

    # 对齐逻辑：保持原始 df 结构
    num_windows = len(all_features)
    window_size = 60
    # 找到每个窗口结束时对应的原始 CSV 行索引
    target_indices = np.arange(num_windows) + window_size - 1
    final_df = full_df.iloc[target_indices].copy().reset_index(drop=True)

    # 拼接特征列 (f_0 到 f_14)
    feat_cols = [f"f_{i}" for i in range(len(all_features[0]))]
    feat_df = pd.DataFrame(all_features, columns=feat_cols)
    final_df = pd.concat([final_df, feat_df], axis=1)

    return final_df


def export_training_data(npy_path, gaf_npy_path, csv_path, trend_weights, reg_weights, device, save_csv_path):
    df = get_model_features(npy_path, gaf_npy_path, csv_path, trend_weights, reg_weights, device)

    # 1. 保存为 CSV (包含 time 列)
    df.to_csv(save_csv_path, index=False)

    # 2. 保存为同名 NPY (剔除 time 列)
    save_npy_path = os.path.splitext(save_csv_path)[0] + ".npy"
    # 明确剔除名为 'time' 的列
    numeric_data = df.drop(columns=['time']).values.astype(np.float32)

    np.save(save_npy_path, numeric_data)

    print(f"导出完成：")
    print(f" - CSV 保存路径: {save_csv_path} (含 time)")
    print(f" - NPY 保存路径: {save_npy_path} (纯数值, 形状: {numeric_data.shape})")


def run(
        NPY_DATA="../CNN/input_x_v1.npy",
        CSV_DATA="../org_v1.csv",
        GAF_DATA="gaf_v1.npy",
        TRAIN_DATA_PATH="ppo_x_v1.csv",
        TREND_WEIGHTS="../CNN_Transformer/transformer_dct_v1.pth",
        REG_WEIGHTS="../CNN/cnn_model_v1.pth",
        COVER=True
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if COVER or not os.path.exists(GAF_DATA):
        precompute_gaf(NPY_DATA, GAF_DATA)

    if COVER or not os.path.exists(TRAIN_DATA_PATH):
        export_training_data(NPY_DATA, GAF_DATA, CSV_DATA, TREND_WEIGHTS, REG_WEIGHTS, DEVICE, TRAIN_DATA_PATH)


if __name__ == "__main__":
    run()