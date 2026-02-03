import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Capability Dashboard", layout="wide")

# CSS để thu nhỏ khoảng cách và kích thước các thẻ
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    [data-testid="stMetricValue"] { font-size: 22px !important; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 10px !important;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #008080;
    }
    .main-header {
        padding: 5px 15px;
        border-radius: 8px;
        background-color: #008080;
        color: white;
        margin-bottom: 15px;
    }
    h1 { font-size: 24px !important; margin-bottom: 0px; }
    h3 { font-size: 18px !important; margin-top: 0px; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING (Giữ nguyên phần kết nối của bạn)
def load_data():
    if "connections" in st.secrets:
        conn = st.connection("gsheets", type=GSheetsConnection)
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        return conn.read(spreadsheet=url, ttl=60)
    return None

df = load_data()

if df is not None:
    # Sidebar thu gọn
    st.sidebar.header("⚙️ 規格設定")
    target_col = st.sidebar.selectbox("數據欄位", df.columns)
    usl = st.sidebar.number_input("USL", value=-0.100, format="%.3f")
    lsl = st.sidebar.number_input("LSL", value=-0.500, format="%.3f")
    
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

        # --- BỐ CỤC MỚI (NHỎ GỌN) ---
        st.markdown('<div class="main-header"><h1>Process Capability Analysis</h1></div>', unsafe_allow_html=True)

        # Chia màn hình làm 2 phần: Trái (Chỉ số) - Phải (Biểu đồ)
        col_left, col_right = st.columns([1, 2.5])

        with col_left:
            st.subheader("📋 Results")
            # Hiển thị dạng lưới nhỏ cho các chỉ số cơ bản
            c1, c2 = st.columns(2)
            c1.metric("N", n)
            c2.metric("Mean", f"{mean:.3f}")
            c3, c4 = st.columns(2)
            c3.metric("Std", f"{std:.3f}")
            c4.metric("Cp", f"{cp:.2f}")
            
            st.write("") # Khoảng cách nhỏ
            
            # Làm nổi bật Cpk
            cpk_color = "#27ae60" if cpk >= 1.33 else "#e67e22" if cpk >= 1.0 else "#e74c3c"
            st.markdown(f"""
                <div style="background-color:{cpk_color}; padding:10px; border-radius:8px; text-align:center; color:white;">
                    <span style="font-size:14px;">Cpk Index</span><br>
                    <span style="font-size:32px; font-weight:bold;">{cpk:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Thêm Ca bên dưới
            st.metric("Ca (Bias)", f"{ca:.2f}")

        with col_right:
            # VẼ BIỂU ĐỒ (Giữ nguyên logic vẽ nhưng chỉnh chiều cao nhỏ lại)
            counts, bins = np.histogram(data, bins=12)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = bins[1] - bins[0]
            bar_colors = ['#ff8a8a' if (x < lsl or x > usl) else '#4a90e2' for x in bin_centers]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=bin_centers, y=counts, width=bin_width*0.9, marker_color=bar_colors, showlegend=False))
            
            x_range = np.linspace(min(data + [lsl]) - 0.1, max(data + [usl]) + 0.1, 200)
            y_curve = stats.norm.pdf(x_range, mean, std) * n * bin_width
            fig.add_trace(go.Scatter(x=x_range, y=y_curve, mode='lines', line=dict(color='black', width=2), name='Normal'))

            # Đường Spec
            fig.add_vline(x=lsl, line_dash="dash", line_color="red")
            fig.add_vline(x=usl, line_dash="dash", line_color="red")

            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=400, # Giảm chiều cao xuống để vừa màn hình
                template="plotly_white",
                xaxis_title="Measurement",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
