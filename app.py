import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION & CSS
st.set_page_config(page_title="QC Analytics", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f5; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border-left: 5px solid #008080;
    }
    .metric-label { color: #64748b; font-size: 13px; font-weight: 600; }
    .metric-value { color: #1e293b; font-size: 20px; font-weight: 700; }
    .cpk-container {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING
def get_data():
    if "connections" in st.secrets:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            return conn.read(spreadsheet=url, ttl=60)
        except Exception as e:
            st.error(f"Connection Error: {e}")
            return None
    return None

df = get_data()

if df is not None:
    # Sidebar: 規格設定
    st.sidebar.header("⚙️ 規格設定")
    target_col = st.sidebar.selectbox("數據欄位", df.columns)
    usl = st.sidebar.number_input("規格上限 (USL)", value=-0.100, format="%.3f")
    lsl = st.sidebar.number_input("規格下限 (LSL)", value=-0.500, format="%.3f")
    
    raw_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
    data = raw_data.tolist()

    if len(data) > 1:
        # CALCULATIONS
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std != 0 else 0
        ca = (mean - (usl + lsl)/2) / ((usl - lsl)/2)

        st.markdown(f'<h2 style="color: #0f172a; margin-bottom:20px;">📊 Process Capability: {target_col} (LAB)</h2>', unsafe_allow_html=True)

        col_metrics, col_chart = st.columns([1, 2.8])

        with col_metrics:
            st.markdown(f"""
                <div class="metric-card"><div class="metric-label">SAMPLE SIZE (N)</div><div class="metric-value">{n}</div></div>
                <div class="metric-card"><div class="metric-label">MEAN / STD DEV</div><div class="metric-value">{mean:.4f} / {std:.4f}</div></div>
                <div class="metric-card"><div class="metric-label">Ca (BIAS) / Cp</div><div class="metric-value">{ca:.2f} / {cp:.2f}</div></div>
            """, unsafe_allow_html=True)

            cpk_bg = "#10b981" if cpk >= 1.33 else "#f59e0b" if cpk >= 1.0 else "#ef4444"
            st.markdown(f"""
                <div class="cpk-container" style="background-color: {cpk_bg};">
                    <div style="font-size: 13px; font-weight: 600; opacity: 0.9;">CPK INDEX</div>
                    <div style="font-size: 36px; font-weight: 900;">{cpk:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_chart:
            # --- CẤU HÌNH BIỂU ĐỒ CÂN ĐỐI ---
            counts, bins = np.histogram(data, bins=12)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = bins[1] - bins[0]
            bar_colors = ['#f87171' if (x < lsl or x > usl) else '#3b82f6' for x in bin_centers]

            fig = go.Figure()

            # Bars
            fig.add_trace(go.Bar(
                x=bin_centers, y=counts, width=bin_width*0.85,
                marker=dict(color=bar_colors, line=dict(color='white', width=1)),
                showlegend=False
            ))

            # Kéo dài đường Normal Curve cân đối hai bên (±4 sigma)
            x_min = min(min(data), lsl) - (2 * std)
            x_max = max(max(data), usl) + (2 * std)
            x_curve = np.linspace(x_min, x_max, 500)
            y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
            
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, 
                mode='lines', 
                line=dict(color='#0f172a', width=2.5), 
                name='Normal Curve'
            ))

            # Spec Lines
            fig.add_vline(x=lsl, line_dash="dash", line_color="#ef4444", line_width=2)
            fig.add_vline(x=usl, line_dash="dash", line_color="#ef4444", line_width=2)

            # Cấu hình khung bao quanh (Box) và lưới
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=450,
                template="plotly_white",
                # Tạo khung bao quanh biểu đồ
                xaxis=dict(
                    title="Measurement Value",
                    mirror=True, # Hiện đường kẻ ở cả 4 cạnh
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='#e2e8f0',
                    range=[x_min, x_max] # Cố định dải hiển thị cân đối
                ),
                yaxis=dict(
                    title="Frequency",
                    mirror=True,
                    ticks='outside',
                    showline=True,
                    linecolor='black',
                    gridcolor='#e2e8f0'
                ),
                legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top', bordercolor="black", borderwidth=1)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
