import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="QC Power BI Dashboard", layout="wide")

# CSS giữ nguyên phong cách Power BI
st.markdown("""
    <style>
    .stApp { background-color: #F3F2F1; }
    .pbi-header { background-color: #004E8C; color: white; padding: 15px 25px; border-radius: 5px; margin-bottom: 20px; }
    .kpi-card { background-color: white; border-radius: 4px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-bottom: 4px solid #004E8C; text-align: center; }
    .kpi-label { color: #605E5C; font-size: 11px; font-weight: 600; text-transform: uppercase; }
    .kpi-value { color: #323130; font-size: 22px; font-weight: 700; }
    .chart-container { background-color: white; padding: 15px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING
def load_data():
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        return conn.read(spreadsheet=url, ttl=60)
    except: return None

df = load_data()

if df is not None:
    # Sidebar
    with st.sidebar:
        st.header("⚙️ CONFIGURATION")
        target_col = st.selectbox("Data Column", df.columns)
        usl = st.number_input("USL", value=1.20)
        lsl = st.number_input("LSL", value=0.70)
        st.info("💡 Tip: Di chuột vào biểu đồ và nhấn biểu tượng 📷 để tải hình ảnh.")

    # Xử lý dữ liệu và tính toán (Giống bản trước)
    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    data = df_clean[target_col].tolist()
    
    if len(data) > 1:
        n, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
        ucl, lcl = mean + 3*std, mean - 3*std
        plot_min, plot_max = min(lsl, lcl, min(data)) - 0.1, max(usl, ucl, max(data)) + 0.1

        # Giao diện chính
        st.markdown(f'<div class="pbi-header"><span style="font-size: 22px; font-weight: 700;">QC ANALYSIS REPORT</span></div>', unsafe_allow_html=True)
        
        # ... (Phần KPI giữ nguyên) ...

        # Vẽ biểu đồ với tính năng tải hình
        col_hist, col_box = st.columns(2)
        
        # Cấu hình chung cho việc tải ảnh
        config_img = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'Quality_Control_Chart',
                'scale': 2 # Ảnh nét gấp đôi màn hình
            }
        }

        with col_hist:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # (Phần tạo fig_hist giữ nguyên)
            # ...
            st.plotly_chart(fig_hist, use_container_width=True, config=config_img)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_box:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # (Phần tạo fig_box giữ nguyên)
            # ...
            st.plotly_chart(fig_box, use_container_width=True, config=config_img)
            st.markdown('</div>', unsafe_allow_html=True)

        # Trend Chart phía dưới
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # (Phần tạo fig_trend giữ nguyên)
        # ...
        st.plotly_chart(fig_trend, use_container_width=True, config=config_img)
        st.markdown('</div>', unsafe_allow_html=True)
