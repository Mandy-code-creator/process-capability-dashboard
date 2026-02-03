import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import math

# 1. 頁面配置與 CSS 樣式
st.set_page_config(page_title="QC 品質控管分析", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F3F2F1; }
    .block-container { padding-top: 1.2rem; }
    .pbi-header {
        background-color: #004E8C; color: white; padding: 12px 20px;
        border-radius: 4px; margin-bottom: 15px;
    }
    .kpi-card {
        background-color: white; border-radius: 4px; padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-bottom: 3px solid #004E8C;
        text-align: center;
    }
    .kpi-label { color: #605E5C; font-size: 10px; font-weight: 600; text-transform: uppercase; }
    .kpi-value { color: #323130; font-size: 20px; font-weight: 700; }
    .chart-container {
        background-color: white; padding: 10px; border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 數據讀取
def load_data():
    if "connections" in st.secrets:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            return conn.read(spreadsheet=url, ttl=60)
        except Exception as e:
            st.error(f"數據讀取失敗: {e}")
            return None
    return None

df = load_data()

if df is not None:
    # --- 側邊欄配置 ---
    with st.sidebar:
        st.header("⚙️ 參數設定")
        target_col = st.selectbox("選擇量測數據欄位", df.columns)
        sigma_val = st.slider("控制界限 Sigma (σ)", min_value=1.0, max_value=4.0, value=3.0, step=0.5)
        
        st.write("---")
        custom_x_label = st.text_input("趨勢圖 X 軸名稱", value="樣本序號")
        y_label = st.text_input("數值單位 (Y 軸)", value="量測值")
        
        st.write("---")
        usl = st.number_input("規格上限 (USL)", value=1.200, format="%.3f")
        lsl = st.number_input("規格下限 (LSL)", value=0.700, format="%.3f")
        
        if st.button("🔄 刷新數據"):
            st.cache_data.clear()
            st.rerun()

    # 數據處理
    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    data = df_clean[target_col].tolist()

    if len(data) > 1:
        # --- 核心統計計算 ---
        n, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
        
        # Sturges 公式計算 Bin 數量
        sturges_k = int(1 + 3.322 * math.log10(n))
        
        # Ca, Cp, Cpk 計算
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        u_center = (usl + lsl) / 2
        tolerance = usl - lsl
        ca = (mean - u_center) / (tolerance / 2) if tolerance != 0 else 0
        cpk = cp * (1 - abs(ca))
        
        ucl, lcl = mean + (sigma_val * std), mean - (sigma_val * std)
        
        # 繪圖範圍優化
        plot_min = min(lsl, lcl, min(data), mean - 3.5*std) - 0.05
        plot_max = max(usl, ucl, max(data), mean + 3.5*std) + 0.05

        # --- 主界面 ---
        st.markdown('<div class="pbi-header"><span style="font-size: 18px; font-weight: 700;">品質控管分析簡報 (QC Performance Report)</span></div>', unsafe_allow_html=True)

        # KPI 概覽列
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        metrics = [
            ("樣本數 (N)", n), 
            ("平均值 (Mean)", f"{mean:.4f}"), 
            ("標準差 (Std)", f"{std:.4f}"),
            ("Ca (準確度)", f"{ca:.2f}"),
            ("Cp (精密度)", f"{cp:.2f}"), 
            ("Cpk (能力指數)", f"{cpk:.2f}")
        ]
        cols = [k1, k2, k3, k4, k5, k6]
        for i, (label, val) in enumerate(metrics):
            cols[i].markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.write("")
        config_dl = {'toImageButtonOptions': {'format': 'png', 'scale': 3}} # 高解析度導出

        # --- ROW 1: DISTRIBUTION (精簡長度版) ---
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        counts, bins = np.histogram(data, bins=sturges_k)
        bin_centers, bin_width = 0.5 * (bins[:-1] + bins[1:]), bins[1] - bins[0]
        bar_colors = ['#FF4B4B' if (x < lsl or x > usl) else '#0078D4' for x in bin_centers]
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, marker_color=bar_colors, name="次數"))
        
        x_curve = np.linspace(plot_min, plot_max, 500)
        y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
        fig_hist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='#333', width=2), name="常態曲線"))
        
        # 標註 USL/LSL
        fig_hist.add_vline(x=usl, line_dash="dash", line_color="#D83B01", line_width=1.5)
        fig_hist.add_vline(x=lsl, line_dash="dash", line_color="#D83B01", line_width=1.5)

        fig_hist.update_layout(
            height=280, # 縮短高度，適合報告呈現
            margin=dict(l=10, r=10, t=35, b=10), 
            template="plotly_white", 
            title=dict(text=f"數據分佈與規格檢核 (Sturges Bins: {sturges_k})", font=dict(size=14)),
            showlegend=False,
            xaxis=dict(range=[plot_min, plot_max], title=y_label, mirror=True, showline=True, linecolor='black'),
            yaxis=dict(title="頻率", mirror=True, showline=True, linecolor='black')
        )
        st.plotly_chart(fig_hist, use_container_width=True, config=config_dl)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ROW 2: TREND CHART ---
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        x_axis = list(range(1, n + 1))
        p_colors = ['#FF4B4B' if (v < lsl or v > usl) else '#0078D4' for v in data]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', 
                                     marker=dict(color=p_colors, size=8, line=dict(width=1, color='white')), 
                                     line=dict(color='#0078D4', width=1.5)))
        
        # 規格與控制界限
        fig_trend.add_hline(y=usl, line_dash="dash", line_color="#D83B01", annotation_text="USL")
        fig_trend.add_hline(y=lsl, line_dash="dash", line_color="#D83B01", annotation_text="LSL")
        fig_trend.add_hline(y=ucl, line_dash="dot", line_color="#107C10", annotation_text=f"UCL ({sigma_val}σ)")
        fig_trend.add_hline(y=lcl, line_dash="dot", line_color="#107C10", annotation_text=f"LCL ({sigma_val}σ)")

        fig_trend.update_layout(
            height=380, margin=dict(l=40, r=40, t=40, b=40), 
            template="plotly_white", 
            title=dict(text=f"生產製程趨勢監控 (±{sigma_val} Sigma)", font=dict(size=14)),
            xaxis=dict(title=custom_x_label, mirror=True, showline=True, linecolor='black'),
            yaxis=dict(title=y_label, mirror=True, showline=True, linecolor='black', range=[plot_min, plot_max])
        )
        st.plotly_chart(fig_trend, use_container_width=True, config=config_dl)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- ROW 3: DATA TABLE ---
        st.markdown('<div style="color: #004E8C; font-weight: 600; margin-bottom: 5px;">📋 詳細量測日誌與判定</div>', unsafe_allow_html=True)
        df_clean['判定 (Result)'] = df_clean[target_col].apply(lambda x: '❌ 不合格' if (x < lsl or x > usl) else '✅ 合格')
        st.dataframe(df_clean, use_container_width=True, hide_index=True)
