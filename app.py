import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION & CSS (Power BI Style - Fixed Header)
st.set_page_config(page_title="QC Power BI Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Nền xám nhạt */
    .stApp { background-color: #F3F2F1; }
    
    /* Căn chỉnh lại khoảng cách toàn trang để không bị che tiêu đề */
    .block-container { 
        padding-top: 2rem !important; 
        padding-bottom: 0rem !important; 
    }
    
    /* Header dải màu xanh đậm chuyên nghiệp - Đảm bảo không bị che */
    .pbi-header {
        background-color: #004E8C; 
        color: white; 
        padding: 20px 25px;
        border-radius: 5px; 
        margin-bottom: 25px; 
        display: flex;
        justify-content: space-between; 
        align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    
    /* Thẻ chỉ số (KPI Cards) */
    .kpi-card {
        background-color: white; border-radius: 4px; padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-bottom: 4px solid #004E8C;
        text-align: center;
    }
    .kpi-label { color: #605E5C; font-size: 12px; font-weight: 600; text-transform: uppercase; }
    .kpi-value { color: #323130; font-size: 22px; font-weight: 700; }
    
    /* Vùng chứa biểu đồ */
    .chart-container {
        background-color: white; padding: 15px; border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 15px;
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
            st.error(f"Error: {e}")
            return None
    return None

df = get_data()

if df is not None:
    # Sidebar - 規格設定
    with st.sidebar:
        st.header("⚙️ 規格設定")
        target_col = st.selectbox("數據欄位", df.columns)
        time_col = st.selectbox("時間軸/批次欄位", [None] + list(df.columns))
        usl = st.number_input("USL", value=-0.100, format="%.3f")
        lsl = st.number_input("LSL", value=-0.500, format="%.3f")
        if st.button("🔄 刷新數據"):
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

        # --- GIAO DIỆN CHÍNH ---
        # Sử dụng tiêu đề với padding đủ lớn để tránh bị che
        st.markdown(f'''
            <div class="pbi-header">
                <span style="font-size: 24px; font-weight: 700;">Quality Control Report: {target_col}</span>
                <span style="font-size: 14px; opacity: 0.8;">Data Source: Google Sheets</span>
            </div>
        ''', unsafe_allow_html=True)

        # Hàng KPI Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        metrics = [("Samples (N)", n), ("Mean", f"{mean:.4f}"), ("StdDev", f"{std:.4f}"), ("Cp", f"{cp:.2f}"), ("Cpk", f"{cpk:.2f}")]
        cols = [k1, k2, k3, k4, k5]
        
        for i, (label, val) in enumerate(metrics):
            cols[i].markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.write("")

        # Bố cục biểu đồ
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # Trend Chart
            x_axis = df_clean[time_col] if time_col else list(range(1, n + 1))
            colors = ['#D83B01' if (v < lsl or v > usl) else '#0078D4' for v in data]
            fig_ctrl = go.Figure()
            fig_ctrl.add_trace(go.Scatter(x=x_axis, y=data, mode='lines+markers', marker=dict(color=colors, size=10), line=dict(color='#0078D4'), name="LAB"))
            fig_ctrl.add_hline(y=usl, line_dash="dash", line_color="#D83B01", annotation_text="USL")
            fig_ctrl.add_hline(y=lsl, line_dash="dash", line_color="#D83B01", annotation_text="LSL")
            fig_ctrl.update_layout(height=450, template="plotly_white", title="Process Trend Analysis", margin=dict(l=40,r=40,t=40,b=40))
            st.plotly_chart(fig_ctrl, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            # Boxplot
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=data, marker_color='#0078D4', boxpoints='all', name="Data"))
            fig_box.update_layout(height=210, margin=dict(l=10,r=10,t=30,b=10), template="plotly_white", title="Boxplot Distribution")
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Histogram
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            counts, bins = np.histogram(data, bins=10)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bar_colors = ['#D83B01' if (x < lsl or x > usl) else '#0078D4' for x in bin_centers]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, marker_color=bar_colors))
            fig_hist.update_layout(height=210, margin=dict(l=10,r=10,t=30,b=10), template="plotly_white", title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
# --- HISTOGRAM VỚI ĐƯỜNG NORMAL CURVE (PHONG CÁCH POWER BI) ---
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # 1. Tính toán Histogram
            counts, bins = np.histogram(data, bins=10)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_width = bins[1] - bins[0]
            
            # Màu sắc cột: Đỏ nếu ngoài Spec, Xanh nếu trong Spec
            bar_colors = ['#D83B01' if (x < lsl or x > usl) else '#0078D4' for x in bin_centers]
            
            fig_hist = go.Figure()

            # Vẽ các cột Histogram
            fig_hist.add_trace(go.Bar(
                x=bin_centers, 
                y=counts, 
                marker_color=bar_colors,
                name="Frequency",
                showlegend=False
            ))

            # 2. Tính toán đường Normal Curve
            # Tạo dải x rộng hơn một chút để đường cong mềm mại (±3 sigma)
            x_min_curve = min(data + [lsl]) - (0.5 * std)
            x_max_curve = max(data + [usl]) + (0.5 * std)
            x_range = np.linspace(x_min_curve, x_max_curve, 200)
            
            # Tính toán PDF và nhân với (Tổng mẫu * Độ rộng bin) để khớp với trục Y của Histogram
            y_normal = stats.norm.pdf(x_range, mean, std) * n * bin_width
            
            # Vẽ đường Normal Curve
            fig_hist.add_trace(go.Scatter(
                x=x_range, 
                y=y_normal, 
                mode='lines', 
                line=dict(color='#323130', width=2), # Màu xám đậm chuyên nghiệp
                name="Normal Curve"
            ))

            # Thêm đường Spec USL/LSL để đối chiếu trực quan
            fig_hist.add_vline(x=usl, line_dash="dot", line_color="#D83B01", line_width=1)
            fig_hist.add_vline(x=lsl, line_dash="dot", line_color="#D83B01", line_width=1)

            fig_hist.update_layout(
                height=250, 
                margin=dict(l=10, r=10, t=30, b=10), 
                template="plotly_white", 
                title="Distribution & Normal Curve",
                xaxis=dict(showline=True, linecolor='#605E5C'),
                yaxis=dict(showgrid=True, gridcolor='#F3F2F1'),
                showlegend=False
            )
            
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
