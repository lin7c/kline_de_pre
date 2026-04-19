import subprocess
import os
import sys
import time
import shutil
from datetime import datetime

# --- 配置 ---
SYMBOLS = [
    "ETHUSDm", "XAUUSDm", "XAGUSDm", "XCUUSDm",
    "UKOILm", "US30m", "USTECm", "HK50m", "DE30m"
]

FILES = ["org_v1.csv", "org_v1.npy"]
DATA_DIR = "data"  # 所有数据存放的子目录


def run_script(script_path, description, arg=None):
    abs_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(abs_path)
    script_name = os.path.basename(abs_path)

    print(f"\n--- 执行: {description} ---")
    cmd = [sys.executable, script_name]
    if arg: cmd.append(arg)

    try:
        subprocess.run(cmd, cwd=script_dir, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ 运行失败: {description}")
        return False


def setup_env():
    """初始化环境：创建 data 目录"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 已创建数据目录: {DATA_DIR}")


def main():
    setup_env()

    # 训练流水线
    pipeline = [
        ("CNN/makedata.py", "CNN 数据预处理"),
        ("RL/makedata.py", "强化学习数据预处理"),
        ("CNN/train.py", "CNN 模型训练"),
        ("CNN_Transformer/makedata.py", "CNN-Transformer 预处理"),
        ("CNN_Transformer/train.py", "CNN-Transformer 训练"),
        ("CNN_Transformer/gy.py", "UD 输入数据"),
        ("UD/makedata.py", "UD 预处理"),
        ("UD/train.py", "UD 训练"),
        ("RL/NPM.py", "PPO 强化学习训练")
    ]

    # --- 阶段 1: 批量数据采集并存入 data/ ---
    print("\n" + "=" * 30 + " 阶段 1: 批量数据采集 " + "=" * 30)
    for symbol in SYMBOLS:
        if run_script("getdata_full.py", f"获取 {symbol} 数据", arg=symbol):
            # 获取成功后，立即移动到 data/ 目录并改名
            for f in FILES:
                if os.path.exists(f):
                    target_path = os.path.join(DATA_DIR, f"{symbol}_{f}")
                    # 如果已存在旧文件则覆盖
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.rename(f, target_path)
                    print(f"📦 数据存入仓库: {target_path}")
        else:
            print(f"🚨 {symbol} 数据获取失败，程序退出。")
            return

    # --- 阶段 2: 轮转训练流水线 ---
    print("\n" + "=" * 30 + " 阶段 2: 轮转训练流水线 " + "=" * 30)

    for i, current_symbol in enumerate(SYMBOLS):
        print(f"\n{'#' * 60}")
        print(f"🔥 当前训练品种: {current_symbol} ({i + 1}/{len(SYMBOLS)})")
        print(f"{'#' * 60}")

        # 1. 【准备阶段】从 data/ 提取数据到根目录供脚本使用
        print(f"🚚 正在提取 {current_symbol} 数据进行训练...")
        prepared = True
        for f in FILES:
            source_path = os.path.join(DATA_DIR, f"{current_symbol}_{f}")
            if os.path.exists(source_path):
                # 使用 copy 而非 rename，防止训练意外中断导致 data 里的原始数据丢失
                shutil.copy2(source_path, f)
            else:
                print(f"⚠️ 找不到源文件: {source_path}")
                prepared = False

        if not prepared:
            print(f"⏭️ {current_symbol} 数据不完整，跳过。")
            continue

        # 2. 执行流水线
        total_start = time.time()
        success = True
        for script, desc in pipeline:
            if not run_script(script, desc, arg=current_symbol):
                success = False
                break

        # 3. 【清理阶段】训练完后删除根目录的临时 org_v1 文件，防止干扰
        for f in FILES:
            if os.path.exists(f):
                os.remove(f)

        if success:
            duration = (time.time() - total_start) / 60
            print(f"\n✨ {current_symbol} 处理完成！耗时: {duration:.2f} 分钟")

        time.sleep(2)

    print("\n✅ 所有品种流水线处理完毕。根目录已清理。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 已手动停止。")