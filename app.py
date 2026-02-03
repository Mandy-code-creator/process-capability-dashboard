import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="QC Analytics Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f5; }
    .metric-card {
        background-color: #ffffff; padding: 12px; border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 10px;
        border-left: 5px solid #008080;
    }
    .metric-label { color: #64748b; font-size: 13px; font-weight: 600; }
    .metric-value { color: #1e293b; font-size: 20px; font-weight: 700; }
    .cpk-container {
        padding: 15px; border-radius: 10px; text-align: center;
        color: white; margin-top: 5px; margin-bottom: 10px;
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
            st.error(f"Connection Error: {e}")
            return None
    return None

df = load_data()

if df is not None:
    # --- SIDEBAR: 規格設定 ---
    st.sidebar.header("⚙️ 規格設定")
    target_col = st.sidebar.selectbox("數據欄位 (Data Column)", df.columns)
    
    # Label cho trục X của Control Chart (ví dụ cột Ngày hoặc ID lô)
    time_col = st.sidebar.selectbox("時間/批次欄位 (Time/Batch Column)", [None] + list(df.columns))
    
    custom_x_label = st.sidebar.text_input("圖表標籤 (X-axis Label)", value=f"{target_col}")
    
    usl = st.sidebar.number_input("USL", value=-0.100, format="%.3f")
    lsl = st.sidebar.number_input("LSL", value=-0.500, format="%.3f")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Xử lý dữ liệu
    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    data = df_clean[target_col].tolist()

    if len(data) > 1:
        # TÍNH TOÁN
        n, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std != 0 else 0
        ca = (mean - (usl + lsl)/2) / ((usl - lsl)/2)

        st.markdown(f'<h2 style="color: #0f172a;">📊 QC Analysis: {target_col}</h2>', unsafe_allow_html=True)

        # BỐ CỤC CHÍNH
        col_metrics, col_charts = st.columns([1, 3])

        with col_metrics:
            st.markdown(f"""
                <div class="metric-card"><div class="metric-label">N</div><div class="metric-value">{n}</div></div>
                <div class="metric-card"><div class="metric-label">MEAN</div><div class="metric-value">{mean:.4f}</div></div>
                <div class="metric-card"><div class="metric-label">STD DEV</div><div class="metric-value">{std:.4f}</div></div>
            """, unsafe_allow_html=True)
            
            cpk_bg = "#10b981" if cpk >= 1.33 else "#f59e0b" if cpk >= 1.0 else "#ef4444"
            st.markdown(f'<div class="cpk-container" style="background-color:{cpk_bg};"><h3>CPK: {cpk:.2f}</h3></div>', unsafe_allow_html=True)
            
            # BOXPLOT NHỎ GỌN
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=data, name="", marker_color='#008080', boxpoints='all'))
            fig_box.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), title="Boxplot Distribution")
            st.plotly_chart(fig_box, use_container_width=True)

        with col_charts:
            tab1, tab2 = st.tabs(["📈 Control Chart (Trend)", "📊 Distribution (Histogram)"])

            with tab1:
                # --- BIỂU ĐỒ KIỂM SOÁT (CONTROL CHART) ---
                fig_control = go.Figure()
                x_axis = df_clean[time_col] if time_col else list(range(1, n + 1))
                
                # Đường dữ liệu thực tế
                fig_control.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', name=f'LAB Data', line=dict(color='#3b82f6')))
                
                # Đường USL/LSL
                fig_control.add_hline(y=usl, line_dash="dash", line_color="red", annotation_text="USL")
                fig_control.add_hline(y=lsl, line_dash="dash", line_color="red", annotation_text="LSL")
                
                # Đường Trung bình (CL)
                fig_control.add_hline(y=mean, line_color="green", annotation_text="Mean")

                fig_control.update_layout(
                    height=450, template="plotly_white", title="Process Control Chart (Trend)",
                    xaxis_title=time_col if time_col else "Index", yaxis_title=custom_x_label,
                    xaxis=dict(mirror=True, showline=True, linecolor='black'),
                    yaxis=dict(mirror=True, showline=True, linecolor='black')
                )
                st.plotly_chart(fig_control, use_container_width=True)

            with tab2:
                # --- HISTOGRAM (GIỮ NGUYÊN BỐ CỤC BOXED CŨ) ---
                counts, bins = np.histogram(data, bins=12)
                bin_centers, bin_width = 0.5 * (bins[:-1] + bins[1:]), bins[1] - bins[0]
                bar_colors = ['#f87171' if (x < lsl or x > usl) else '#4a90e2' for x in bin_centers]

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, width=bin_width*0.85, marker_color=bar_colors, showlegend=False))
                
                x_curve = np.linspace(min(data + [lsl]) - 0.1, max(data + [usl]) + 0.1, 200)
                y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
                fig_hist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='black', width=2), name='Normal'))

                fig_hist.update_layout(
                    height=450, template="plotly_white", xaxis_title=custom_x_label, yaxis_title="Frequency",
                    xaxis=dict(mirror=True, showline=True, linecolor='black'),
                    yaxis=dict(mirror=True, showline=True, linecolor='black')
                )
                st.plotly_chart(fig_hist, use_container_width=True)
