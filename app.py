import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import math

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="QC 品質控管分析報告", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .pbi-header {
        background-color: #004E8C; color: white; padding: 10px 20px;
        border-radius: 4px; margin-bottom: 10px;
    }
    .kpi-card {
        background-color: white; border-radius: 4px; padding: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); border-bottom: 3px solid #004E8C;
        text-align: center;
    }
    .kpi-label { color: #605E5C; font-size: 10px; font-weight: 600; }
    .kpi-value { color: #323130; font-size: 18px; font-weight: 700; }
    .chart-container {
        background-color: white; padding: 10px; border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING
def load_data():
    if "connections" in st.secrets:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            return conn.read(spreadsheet=url, ttl=60)
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    return None

df = load_data()

if df is not None:
    with st.sidebar:
        st.header("⚙️ 參數設定")
        target_col = st.selectbox("選擇量測欄位", df.columns)
        sigma_val = st.slider("控制界限 Sigma (σ)", 1.0, 4.0, 3.0, 0.5)
        
        st.write("---")
        custom_x_label = st.text_input("趨勢圖 X 軸標籤", "樣本序號")
        y_label = st.text_input("單位/數值標籤", "量測值")
        
        st.write("---")
        usl = st.number_input("規格上限 (USL)", value=1.200, format="%.3f")
        lsl = st.number_input("規格下限 (LSL)", value=0.700, format="%.3f")
        
        if st.button("🔄 刷新數據"):
            st.cache_data.clear()
            st.rerun()

    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    data = df_clean[target_col].tolist()

    if len(data) > 1:
        # --- CALCULATIONS ---
        n, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
        sturges_k = int(1 + 3.322 * math.log10(n))
        
        u_spec = (usl + lsl) / 2
        t_spec = usl - lsl
        ca = (mean - u_spec) / (t_spec / 2) if t_spec != 0 else 0
        cp = t_spec / (6 * std) if std != 0 else 0
        cpk = cp * (1 - abs(ca))
        
        ucl, lcl = mean + (sigma_val * std), mean - (sigma_val * std)
        plot_min = min(lsl, lcl, min(data), mean - 3.5*std) - 0.05
        plot_max = max(usl, ucl, max(data), mean + 3.5*std) + 0.05

        # --- MAIN UI ---
        st.markdown(f'<div class="pbi-header"><span style="font-size: 16px; font-weight: 700;">品質分析報告 | {target_col}</span></div>', unsafe_allow_html=True)

        # KPI Row
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        metrics = [("樣本數", n), ("平均值 μ", f"{mean:.3f}"), ("標準差 σ", f"{std:.3f}"), 
                   ("Ca (準確度)", f"{ca:.2f}"), ("Cp (精密度)", f"{cp:.2f}"), ("Cpk (能力)", f"{cpk:.2f}")]
        cols = [k1, k2, k3, k4, k5, k6]
        for i, (label, val) in enumerate(metrics):
            cols[i].markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.write("")
        config_dl = {'toImageButtonOptions': {'format': 'png', 'scale': 3, 'filename': 'QC_Export_Parallel'}}

        # --- BIỂU ĐỒ SONG SONG ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            counts, bins = np.histogram(data, bins=sturges_k)
            bin_centers, bin_width = 0.5 * (bins[:-1] + bins[1:]), bins[1] - bins[0]
            bar_colors = ['#FF4B4B' if (x < lsl or x > usl) else '#0078D4' for x in bin_centers]
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, marker_color=bar_colors, opacity=0.7))
            x_curve = np.linspace(plot_min, plot_max, 500)
            y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
            fig_hist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='red', width=2)))
            
            fig_hist.add_vline(x=usl, line_dash="dash", line_color="#D83B01")
            fig_hist.add_vline(x=lsl, line_dash="dash", line_color="#D83B01")

            fig_hist.update_layout(
                height=300, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white",
                title=dict(text=f"分佈概覽 (μ={mean:.2f})", font=dict(size=12)),
                showlegend=False,
                xaxis=dict(range=[plot_min, plot_max], title=y_label, mirror=True, showline=True, linecolor='black'),
                yaxis=dict(title="頻率", mirror=True, showline=True, linecolor='black')
            )
            st.plotly_chart(fig_hist, use_container_width=True, config=config_dl)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            x_axis = list(range(1, n + 1))
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', 
                                         marker=dict(color=['#FF4B4B' if (v < lsl or v > usl) else '#0078D4' for v in data], size=6), 
                                         line=dict(color='#0078D4', width=1.5)))
            
            fig_trend.add_hline(y=usl, line_dash="dash", line_color="#D83B01")
            fig_trend.add_hline(y=lsl, line_dash="dash", line_color="#D83B01")
            fig_trend.add_hline(y=ucl, line_dash="dot", line_color="#107C10")
            fig_trend.add_hline(y=lcl, line_dash="dot", line_color="#107C10")

            fig_trend.update_layout(
                height=300, margin=dict(l=40, r=40, t=30, b=40), template="plotly_white",
                title=dict(text="趨勢監控圖", font=dict(size=12)),
                xaxis=dict(title=custom_x_label, mirror=True, showline=True, linecolor='black'),
                yaxis=dict(title=y_label, mirror=True, showline=True, linecolor='black', range=[plot_min, plot_max])
            )
            st.plotly_chart(fig_trend, use_container_width=True, config=config_dl)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- ROW 3: DATA TABLE ---
        st.markdown('<div style="color: #004E8C; font-weight: 600; margin-bottom: 5px;">📋 詳細量測紀錄</div>', unsafe_allow_html=True)
        df_clean['判定'] = df_clean[target_col].apply(lambda x: '❌ OUT' if (x < lsl or x > usl) else '✅ PASS')
        st.dataframe(df_clean, use_container_width=True, hide_index=True)
