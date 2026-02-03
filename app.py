import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. CẤU HÌNH TRANG & CSS NÂNG CAO
st.set_page_config(page_title="QC Analytics", layout="wide")

st.markdown("""
    <style>
    /* Tổng thể nền xám nhạt hiện đại */
    .stApp { background-color: #f0f2f5; }
    
    /* Thu nhỏ khoảng cách giữa các phần */
    .block-container { padding-top: 1.5rem; padding-bottom: 0px; }
    
    /* Thiết kế các thẻ Card trắng sang trọng */
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 5px solid #008080;
    }
    
    .metric-label { color: #64748b; font-size: 14px; font-weight: 600; margin-bottom: 5px; }
    .metric-value { color: #1e293b; font-size: 24px; font-weight: 700; }
    
    /* Khối Cpk nổi bật */
    .cpk-container {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-top: 10px;
    }
    
    /* Tiêu đề chính */
    .main-title {
        color: #0f172a;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. KẾT NỐI DỮ LIỆU
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
    # Sidebar: 規格設定 (Bản Phồn thể)
    st.sidebar.header("⚙️ 規格設定")
    target_col = st.sidebar.selectbox("數據欄位", df.columns)
    usl = st.sidebar.number_input("規格上限 (USL)", value=-0.100, format="%.3f")
    lsl = st.sidebar.number_input("規格下限 (LSL)", value=-0.500, format="%.3f")
    
    raw_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
    data = raw_data.tolist()

    if len(data) > 1:
        # TÍNH TOÁN
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std != 0 else 0
        ca = (mean - (usl + lsl)/2) / ((usl - lsl)/2)

        # --- BỐ CỤC CHÍNH ---
        st.markdown(f'<div class="main-title">📊 Process Capability: {target_col} (LAB)</div>', unsafe_allow_html=True)

        # Chia cột: Cột trái (30%) cho thông số - Cột phải (70%) cho biểu đồ
        col_metrics, col_chart = st.columns([1, 2.5])

        with col_metrics:
            # Tạo các Card thủ công bằng HTML để đẹp hơn st.metric mặc định
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">SAMPLE SIZE (N)</div>
                    <div class="metric-value">{n}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MEAN / STD DEV</div>
                    <div class="metric-value">{mean:.4f} / {std:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ca (BIAS) / Cp</div>
                    <div class="metric-value">{ca:.2f} / {cp:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

            # Cpk Highlight Box
            cpk_bg = "#10b981" if cpk >= 1.33 else "#f59e0b" if cpk >= 1.0 else "#ef4444"
            st.markdown(f"""
                <div class="cpk-container" style="background-color: {cpk_bg};">
                    <div style="font-size: 14px; font-weight: 600; opacity: 0.9;">CPK CAPACITY INDEX</div>
                    <div style="font-size: 48px; font-weight: 900;">{cpk:.2f}</div>
                </div>
            """, unsafe_allow_html=True)

        with col_chart:
            # VẼ BIỂU ĐỒ TỐI ƯU
            counts, bins = np.histogram(data, bins=12)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = bins[1] - bins[0]
            
            # Màu sắc: Xanh đậm và Đỏ cảnh báo
            bar_colors = ['#f87171' if (x < lsl or x > usl) else '#3b82f6' for x in bin_centers]

            fig = go.Figure()

            # Bars
            fig.add_trace(go.Bar(
                x=bin_centers, y=counts, width=bin_width*0.85,
                marker=dict(color=bar_colors, line=dict(color='white', width=1)),
                showlegend=False
            ))

            # Black Curve
            x_curve = np.linspace(min(data + [lsl]) - 0.1, max(data + [usl]) + 0.1, 200)
            y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
            fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='#0f172a', width=3), name='Normal Curve'))

            # Spec Lines
            fig.add_vline(x=lsl, line_dash="dash", line_color="#ef4444", line_width=2)
            fig.add_vline(x=usl, line_dash="dash", line_color="#ef4444", line_width=2)

            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=480,
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)', # Trong suốt để tiệp màu nền
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#e2e8f0', title="Measurement Value"),
                yaxis=dict(gridcolor='#e2e8f0', title="Frequency"),
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
