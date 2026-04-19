import streamlit as st
import json
import time
import os
import pandas as pd
import plotly.graph_objects as go
import hashlib

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="PPO 评估监控",
    layout="wide",
    initial_sidebar_state="collapsed"
)

JSON_PATH = "evaluate_mario_state.json"

ACTION_COLORS = {0: '#00ccff', 1: '#00ff00', 2: '#ff4b4b'}

L2_MAP = {
    0: "观望 ⚪", 1: "开多 🟢", 2: "开空 🔴",
    3: "持多 📈", 4: "持空 📉",
    5: "平多 ✅", 6: "平空 ✅",
}

# 样式
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetric"] {
        background-color: #1f2937 !important;
        padding: 12px;
        border-radius: 8px;
    }
    [data-testid="stMetricLabel"] p { color: #9ca3af !important; font-size: 0.95rem; }
    [data-testid="stMetricValue"] div { color: #ffffff !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 PPO 评估监控仪表盘（简化版）")

placeholder = st.empty()

# 文件变化检测
last_hash = None
last_mtime = 0

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def render_ui():
    global last_hash, last_mtime

    if not os.path.exists(JSON_PATH):
        st.warning(f"等待 {JSON_PATH} 生成中...（请在 evaluate_policy 中设置 show=1）")
        return

    current_mtime = os.path.getmtime(JSON_PATH)
    current_hash = get_file_hash(JSON_PATH)

    if current_hash == last_hash and current_mtime == last_mtime:
        return  # 文件未变化，不刷新

    last_hash = current_hash
    last_mtime = current_mtime

    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data or not isinstance(data, list) or len(data) == 0:
            st.info("数据加载中...")
            return

        latest = data[-1]
        df = pd.DataFrame(data)

        with placeholder.container():
            # ====================== 顶部指标 ======================
            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("全局步数", f"{latest.get('global_step', 0):,}")
            c2.metric("当前净值", f"{latest.get('net_worth', 0):.2f}")
            c3.metric("当前价格", f"{latest.get('price', 0):.4f}")

            act = latest.get('action', 0)
            action_name = ["观望", "开多", "开空"][act] if act in [0,1,2] else "未知"
            c4.metric("执行动作", f"{action_name} ({act})")

            reasonable = latest.get('is_reasonable', False)
            c5.metric("决策合理性", "✅ 合理" if reasonable else "❌ 不合理")

            st.markdown(f"**当前仓位：** { {0:'空仓 ⚪', 1:'持多 🟢', 2:'持空 🔴'}.get(latest.get('pos_after', 0), '未知')}")

            st.markdown("---")

            # ====================== 简化价格图表 ======================
            if len(df) > 1:
                fig = go.Figure()

                # 1. 价格线（单条灰色线，清晰不卡）
                fig.add_trace(go.Scatter(
                    x=df['global_step'],
                    y=df['price'],
                    mode='lines',
                    line=dict(color='#888888', width=2),
                    name='价格',
                    hoverinfo='skip'
                ))

                # 2. 动作标记点（带颜色）
                fig.add_trace(go.Scatter(
                    x=df['global_step'],
                    y=df['price'],
                    mode='markers',
                    marker=dict(
                        color=[ACTION_COLORS.get(int(a), '#ffffff') for a in df['action']],
                        size=8,
                        line=dict(width=1, color='white')
                    ),
                    name='动作点',
                    customdata=list(zip(
                        [L2_MAP.get(int(l2), str(l2)) for l2 in df['pred_l2']],
                        df['reward'].round(4)
                    )),
                    hovertemplate=(
                        '<b>步数:</b> %{x}<br>'
                        '<b>价格:</b> %{y:.4f}<br>'
                        '<b>预测 L2:</b> %{customdata[0]}<br>'
                        '<b>本步奖励:</b> %{customdata[1]:+.4f}<extra></extra>'
                    )
                ))

                fig.update_layout(
                    template="plotly_dark",
                    height=480,                    # 降低高度，更轻量
                    margin=dict(l=10, r=10, t=30, b=10),
                    title="价格走势（点颜色表示执行动作）",
                    xaxis_title="全局步数",
                    yaxis_title="价格",
                    hovermode="closest",
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)

            # ====================== 最新决策详情 ======================
            st.subheader("最新一步决策")
            col_left, col_right = st.columns(2)

            with col_left:
                st.write("**预测 L2：**", L2_MAP.get(latest.get('pred_l2', -1), "未知"))
                st.write("**真实 L2：**", L2_MAP.get(latest.get('true_l2', -1), "未知"))
                st.write("**预测 L3：**", latest.get('pred_l3', 'N/A'))

            with col_right:
                st.write("**动作前仓位：**", latest.get('pos_before', 0))
                st.write("**动作后仓位：**", latest.get('pos_after', 0))
                st.write("**本步奖励：**", f"{latest.get('reward', 0):+.4f}")

            st.caption(f"总记录: {len(data)} 条 | 文件变化时自动更新 | {time.strftime('%H:%M:%S')}")

    except Exception as e:
        st.error(f"读取错误: {str(e)}")


if __name__ == "__main__":
    st.info("✅ 简化版监控已启动（仅文件变化时刷新，图表更流畅）")
    while True:
        render_ui()
        time.sleep(1.0)   # 1秒检测一次，足够流畅且不卡