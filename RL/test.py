import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# ====================== 1. 基础配置与数据加载 ======================
COL_NAMES = [
                "time", "1m_O", "1m_H", "1m_L", "1m_C",
                "5m_O", "5m_H", "5m_L", "5m_C",
                "15m_O", "15m_H", "15m_L", "15m_C"
            ] + [f"f_{i}" for i in range(15)]

L1_MAP = {0: "空仓 (0)", 1: "多头 (1)", 2: "空头 (2)"}
L2_MAP = {
    0: "观望 (0)", 1: "开多 (1)", 2: "开空 (2)",
    3: "持多 (3)", 4: "持空 (4)", 5: "平多 (5)", 6: "平空 (6)"
}
L1_REVERSE = {v: k for k, v in L1_MAP.items()}
L2_REVERSE = {v: k for k, v in L2_MAP.items()}

TRAIN_DATA_PATH="ppo_x_v1.csv"
MODEL_PATH = "ppo_v1",
ANNOTATION_JSON = "ppo_v1_l3_annotation_log_2.json"
@st.cache_data
def load_data(path = TRAIN_DATA_PATH):
    if not os.path.exists(path): return None
    df = pd.read_csv(path, names=COL_NAMES, header=None)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.reset_index(drop=True)


def load_annotations(path = ANNOTATION_JSON):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 转换为以 step 为 key 的字典
                return {int(item['step']): item for item in data}
        except:
            return {}
    return {}


def save_annotations(anno_dict,path = ANNOTATION_JSON):
    save_list = sorted(list(anno_dict.values()), key=lambda x: x['step'])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)


# --- 页面初始化 ---
st.set_page_config(layout="wide", page_title="RL训练标注与收益分析")

if 'df' not in st.session_state:
    st.session_state.df = load_data()
if 'annotations' not in st.session_state:
    st.session_state.annotations = load_annotations()
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

df = st.session_state.df
if df is None:
    st.error("未找到数据源 ppo_x_v1.csv")
    st.stop()

# ====================== 2. 侧边栏：全局控制 ======================
with st.sidebar:
    st.header("⚙️ 全局控制")
    win_size = st.number_input("K线窗口大小", 20, 1000, 100)

    st.divider()
    jump = st.number_input("跳至 Step", 0, len(df) - 1, st.session_state.current_step)
    if st.button("确认跳转"):
        st.session_state.current_step = jump
        st.rerun()

# ====================== 3. 主界面：使用 Tabs 切换 ======================
tab_label, tab_perf = st.tabs(["🎯 数据标注终端", "📈 收益曲线分析"])

# ---------------- Tab 1: 标注终端 ----------------
with tab_label:
    curr_step = st.session_state.current_step
    start_idx = max(0, curr_step - win_size // 2)
    end_idx = min(len(df), start_idx + win_size)
    df_slice = df.iloc[start_idx:end_idx]

    col_chart, col_edit = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_slice.index, open=df_slice['1m_O'], high=df_slice['1m_H'],
            low=df_slice['1m_L'], close=df_slice['1m_C'], name='Market'
        ))
        # 焦点标记
        fig.add_trace(go.Scatter(
            x=[curr_step], y=[df.loc[curr_step, '1m_C']],
            mode='markers', marker=dict(size=18, color="yellow", symbol="star"),
            name="Current Step"
        ))
        fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(l=10, r=10, t=10, b=10))

        event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True,config={'scrollZoom': True})
        if event and "selection" in event and event["selection"]["points"]:
            st.session_state.current_step = int(event["selection"]["points"][0]["x"])
            st.rerun()

    with col_edit:
        step = st.session_state.current_step
        current_anno = st.session_state.annotations.get(step, {
            "step": step, "action_l1": 0, "state_l2": 0, "detail_l3": 0,
            "reward": 0.0, "net_worth": 1000.0, "comment": ""
        })

        st.subheader(f"📝 标注: Step {step}")
        with st.form("edit_form"):
            l1_val = int(current_anno.get("action_l1", 0))
            l2_val = int(current_anno.get("state_l2", 0))

            new_l1_text = st.selectbox("Action L1", options=list(L1_MAP.values()),
                                       index=list(L1_MAP.keys()).index(l1_val) if l1_val in L1_MAP else 0)
            new_l2_text = st.selectbox("State L2", options=list(L2_MAP.values()),
                                       index=list(L2_MAP.keys()).index(l2_val) if l2_val in L2_MAP else 0)
            new_l3 = st.number_input("Detail L3", value=int(current_anno.get("detail_l3", 0)))
            new_comment = st.text_area("评语", value=current_anno.get("comment", ""), height=100)

            # 允许手动修正净值（可选）
            new_nw = st.number_input("Net Worth", value=float(current_anno.get("net_worth", 1000.0)))

            if st.form_submit_button("💾 保存当前修改", use_container_width=True):
                current_anno.update({
                    "action_l1": L1_REVERSE[new_l1_text],
                    "state_l2": L2_REVERSE[new_l2_text],
                    "detail_l3": new_l3,
                    "comment": new_comment,
                    "net_worth": new_nw
                })
                st.session_state.annotations[step] = current_anno
                save_annotations(st.session_state.annotations)
                st.success("已保存")
                st.rerun()

# ---------------- Tab 2: 收益曲线分析 ----------------
with tab_perf:
    st.subheader("📈 策略运行表现 (基于 JSON 标注记录)")

    if not st.session_state.annotations:
        st.warning("JSON 文件中尚无标注数据，无法绘制曲线。")
    else:
        # 将标注字典转为 DataFrame 并按 step 排序
        anno_df = pd.DataFrame(list(st.session_state.annotations.values()))
        anno_df = anno_df.sort_values("step")

        # 创建双轴图表：净值 + 奖励
        fig_perf = make_subplots(specs=[[{"secondary_y": True}]])

        # 净值曲线 (Net Worth)
        fig_perf.add_trace(
            go.Scatter(x=anno_df["step"], y=anno_df["net_worth"],
                       name="Net Worth (净值)", line=dict(color="#00ff00", width=3)),
            secondary_y=False,
        )

        # 奖励分布 (Reward) - 用柱状图或散点
        fig_perf.add_trace(
            go.Bar(x=anno_df["step"], y=anno_df["reward"],
                   name="Reward (奖励)", marker_color="rgba(255, 165, 0, 0.4)"),
            secondary_y=True,
        )

        fig_perf.update_layout(
            title="账户净值与奖励随时间变化图",
            template="plotly_dark",
            height=600,
            xaxis_title="Step (时间步)",
            hovermode="x unified"
        )

        fig_perf.update_yaxes(title_text="<b>Net Worth</b>", secondary_y=False)
        fig_perf.update_yaxes(title_text="<b>Reward</b>", secondary_y=True)

        st.plotly_chart(fig_perf, use_container_width=True)

        # 数据预览表
        with st.expander("查看标注原始数据表"):
            st.dataframe(anno_df, use_container_width=True)