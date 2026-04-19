import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_trading_data(df, features_to_show=['f_0', 'f_3', 'f_6'], window_size=100):
    """
    df: 原始 DataFrame
    features_to_show: 想要显示的特征列表
    window_size: 显示最近的 N 行数据
    """
    # 1. 截取指定长度的数据
    plot_df = df.tail(window_size).copy()

    # 2. 准备归一化数据
    scaler = MinMaxScaler()
    cols = ['close'] + features_to_show
    plot_df[cols] = scaler.fit_transform(plot_df[cols])

    # 3. 绘图
    plt.figure(figsize=(14, 7), dpi=100)

    # 绘制价格曲线 (加粗、黑色、置于顶层)
    plt.plot(plot_df['date'], plot_df['close'],
             label='Close (Normalized)', color='black', linewidth=2.5, zorder=5)

    # 循环绘制选中的特征
    for col in features_to_show:
        plt.plot(plot_df['date'], plot_df[col], label=f'Feature: {col}', alpha=0.7)

    # 格式化图表
    plt.title(f'Comparison of Close and Features (Last {window_size} intervals)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Normalized Value [0, 1]')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例放在外侧避免遮挡
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# --- 使用示例 ---
# 假设 df 是你读取的完整数据
# 查看最近 50 条数据
TRAIN_DATA_PATH= "ppo_x_v1.csv"
df = pd.read_csv(TRAIN_DATA_PATH)
plot_trading_data(df, features_to_show=['f_0','f_1'], window_size=100)

# 查看最近 200 条数据
# plot_trading_data(df, features_to_show=['f_3', 'f_4'], window_size=200)