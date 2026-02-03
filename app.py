import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION & CSS (Power BI Style - GIỮ NGUYÊN)
st.set_page_config(page_title="QC Power BI Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F3F2F1; }
    .block-container { padding-top: 1.5rem; }
    .pbi-header {
        background-color: #004E8C; color: white; padding: 15px 25px;
        border-radius: 5px; margin-bottom: 20px;
    }
    .kpi-card {
        background-color: white; border-radius: 4px; padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-bottom: 4px solid #004E8C;
        text-align: center;
    }
    .kpi-label { color: #605E5C; font-size: 11px; font-weight: 600; text-transform: uppercase; }
    .kpi-value { color: #323130; font-size: 22px; font-weight: 700; }
    .chart-container {
        background-color: white; padding: 15px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING (GIỮ NGUYÊN)
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
        st.header("⚙️ CONFIGURATION")
        target_col = st.selectbox("Data Column", df.columns)
        time_col = st.selectbox("Batch/Time Column", [None] + list(df.columns))
        x_label = st.text_input("X-axis Label", value="Measurement Analysis")
        usl = st.number_input("Upper Spec Limit (USL)", value=1.200, format="%.3f")
        lsl = st.number_input("Lower Spec Limit (LSL)", value=0.700, format="%.3f")
        if st.button("🔄 REFRESH DATA"):
            st.cache_data.clear()
            st.rerun()

    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    data = df_clean[target_col].tolist()

    if len(data) > 1:
        # --- CALCULATIONS ---
        n, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std != 0 else 0
        ucl, lcl = mean + (3 * std), mean - (3 * std)

        plot_min = min(lsl, lcl, min(data), mean - 3.5*std) - 0.1
        plot_max = max(usl, ucl, max(data), mean + 3.5*std) + 0.1

        # --- MAIN UI ---
        st.markdown(f'<div class="pbi-header"><span style="font-size: 22px; font-weight: 700;">QUALITY CONTROL REPORT</span></div>', unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        metrics = [("N", n), ("MEAN", f"{mean:.4f}"), ("STD DEV", f"{std:.4f}"), ("CP", f"{cp:.2f}"), ("CPK", f"{cpk:.2f}")]
        cols = [k1, k2, k3, k4, k5]
        for i, (label, val) in enumerate(metrics):
            cols[i].markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.write("")

        # --- CẤU HÌNH TẢI ẢNH (Dùng chung cho các biểu đồ) ---
        config_download = {
            'toImageButtonOptions': {
                'format': 'png', 
                'filename': 'QC_Dashboard_Export',
                'scale': 2 # Tăng độ nét ảnh khi tải về
            }
        }

        # --- TOP ROW: DISTRIBUTION & BOXPLOT ---
        col_hist, col_box = st.columns(2)

        with col_hist:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            counts, bins = np.histogram(data, bins=10)
            bin_centers, bin_width = 0.5 * (bins[:-1] + bins[1:]), bins[1] - bins[0]
            bar_colors = ['#FF0000' if (x < lsl or x > usl) else '#0078D4' for x in bin_centers]
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, marker_color=bar_colors, name="Freq"))
            
            x_curve = np.linspace(plot_min, plot_max, 500)
            y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
            fig_hist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='black', width=2), name="Normal"))
            
            fig_hist.add_vline(x=usl, line_dash="dash", line_color="#D83B01", line_width=2, annotation_text="USL")
            fig_hist.add_vline(x=lsl, line_dash="dash", line_color="#D83B01", line_width=2, annotation_text="LSL")

            fig_hist.update_layout(
                height=350, margin=dict(l=10,r=10,t=40,b=10), template="plotly_white", 
                title="Distribution & Spec Compliance", showlegend=False,
                xaxis=dict(range=[plot_min, plot_max], title=x_label, mirror=True, showline=True, linecolor='black'),
                yaxis=dict(mirror=True, showline=True, linecolor='black')
            )
            # THÊM CONFIG Ở ĐÂY
            st.plotly_chart(fig_hist, use_container_width=True, config=config_download)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_box:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=data, marker=dict(color='#0078D4', outliercolor='#FF0000'), boxpoints='all', jitter=0.3))
            
            fig_box.add_hline(y=usl, line_dash="dash", line_color="#D83B01", annotation_text="USL")
            fig_box.add_hline(y=lsl, line_dash="dash", line_color="#D83B01", annotation_text="LSL")
            
            fig_box.update_layout(
                height=350, margin=dict(l=10,r=10,t=40,b=10), template="plotly_white", 
                title="Boxplot & Outliers",
                yaxis=dict(range=[plot_min, plot_max], mirror=True, showline=True, linecolor='black')
            )
            # THÊM CONFIG Ở ĐÂY
            st.plotly_chart(fig_box, use_container_width=True, config=config_download)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- BOTTOM ROW: TREND CHART ---
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        x_axis = df_clean[time_col] if time_col else list(range(1, n + 1))
        p_colors = ['#FF0000' if (v < lsl or v > usl) else '#0078D4' for v in data]
        p_sizes = [12 if (v < lsl or v > usl) else 8 for v in data]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', 
                                     marker=dict(color=p_colors, size=p_sizes, line=dict(width=1, color='white')), 
                                     line=dict(color='#0078D4', width=2)))
        
        fig_trend.add_hline(y=usl, line_dash="dash", line_color="#D83B01", annotation_text="USL (Spec)")
        fig_trend.add_hline(y=lsl, line_dash="dash", line_color="#D83B01", annotation_text="LSL (Spec)")
        fig_trend.add_hline(y=ucl, line_dash="dot", line_color="#107C10", annotation_text="UCL (Control)")
        fig_trend.add_hline(y=lcl, line_dash="dot", line_color="#107C10", annotation_text="LCL (Control)")

        fig_trend.update_layout(height=450, margin=dict(l=40,r=40,t=40,b=40), template="plotly_white", 
                               title="Process Trend (Control Chart)",
                               xaxis=dict(mirror=True, showline=True, linecolor='black'),
                               yaxis=dict(mirror=True, showline=True, linecolor='black', range=[plot_min, plot_max]))
        # THÊM CONFIG Ở ĐÂY
        st.plotly_chart(fig_trend, use_container_width=True, config=config_download)
        st.markdown('</div>', unsafe_allow_html=True)
