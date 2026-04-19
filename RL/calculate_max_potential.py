import numpy as np


def calculate_max_potential_sharpe(npy_path, num_trades_re=100):
    # 1. 加载数据
    try:
        data = np.load(npy_path).astype(np.float32)
    except FileNotFoundError:
        return "File Not Found"

    num_trades = len(data) // num_trades_re
    prices = data[:, 3]
    n = len(prices)

    if n < 2:
        return 0.0

    # 2. 动态规划获取最优交易路径下的累计利润序列
    # dp[k][i] 表示在时刻 i，完成最多 k 次交易的最大累计利润
    dp = np.zeros((num_trades + 1, n))

    for k in range(1, num_trades + 1):
        # 记录在第 k 次交易中，买入后的最低净成本
        min_cost = prices[0]
        for i in range(1, n):
            # 更新成本：当前价格 - 之前的累计利润
            min_cost = min(min_cost, prices[i] - dp[k - 1][i - 1])
            # 更新利润：保持上一时刻利润 vs 当前价格卖出的利润
            dp[k][i] = max(dp[k][i - 1], prices[i] - min_cost)

    # 3. 提取收益率序列 (Returns)
    # 最终的最优利润曲线
    optimal_profit_curve = dp[num_trades]
    # 计算每根 K 线的收益变化量，并转化为收益率
    # 注意：这里除以起始价格 prices[0] 或前一时刻价格均可，取决于你的习惯
    # 为了稳定性和避免除零，使用 prices[0] 作为基准
    periodic_returns = np.diff(optimal_profit_curve) / (prices[:-1] + 1e-8)

    # 4. 计算夏普率
    avg_return = np.mean(periodic_returns)
    std_return = np.std(periodic_returns)

    if std_return < 1e-10:
        return 0.0

    sharpe_ratio = avg_return / std_return

    # 如果需要年化夏普率，可以乘以 np.sqrt(N)，N 为一年的 K 线数量
    return sharpe_ratio


if __name__ == "__main__":
    TRAIN_DATA_PATH = "ppo_x_v1.npy"

    result = calculate_max_potential_sharpe(TRAIN_DATA_PATH)
    if isinstance(result, str):
        print(f"错误: {result}")
    else:
        print(f"最大潜力夏普率: {result:.6f}")