import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import math

# --- 1. LỆNH NÀY PHẢI LUÔN ĐỨNG ĐẦU TIÊN ---
# Thiết lập Wide Mode để tận dụng toàn bộ chiều ngang màn hình
st.set_page_config(page_title="QC 品質控管分析報告", layout="wide")

# --- 2. CSS TỐI ƯU HÓA (Đảm bảo không chồng lấn nội dung) ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .block-container { padding-top: 0.5rem; padding-bottom: 1rem; }
    .pbi-header {
        background-color: #004E8C; color: white; padding: 10px 20px;
        border-radius: 4px; margin-bottom: 15px;
    }
    .kpi-card {
        background-color: white; border-radius: 4px; padding: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-bottom: 4px solid #004E8C;
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

# --- 3. DATA LOADING ---
def load_data():
    if "connections" in st.secrets:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            return conn.read(spreadsheet=url, ttl=60)
        except Exception as e:
            st.error(f"數據連接錯誤: {e}")
            return None
    return None

df = load_data()

if df is not None:
    # --- SIDEBAR: THIẾT LẬP THÔNG SỐ ---
    with st.sidebar:
        st.header("⚙️ 參數設定")
        target_col = st.selectbox("選擇量測數據欄位", df.columns)
        
        # Nhập Sigma
        sigma_val = st.number_input("控制界限 Sigma (σ)", min_value=0.1, max_value=6.0, value=3.0, step=0.1)
        
        st.write("---")
        # QUAN TRỌNG: Thiết lập USL/LSL mặc định là 65 và 55 để khớp với dữ liệu thực tế
        usl = st.number_input("規格上限 (USL)", value=65.0, step=1.0)
        lsl = st.number_input("規格下限 (LSL)", value=55.0, step=1.0)
        
        st.write("---")
        custom_x_label = st.text_input("X 軸標籤", "樣本序號")
        y_label = st.text_input("Y 軸標籤", "量測值 (HRB)")
        
        if st.button("🔄 刷新數據"):
            st.cache_data.clear()
            st.rerun()

    # Xử lý dữ liệu
    df_clean = df.copy()
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
    df_clean = df_clean.dropna(subset=[target_col])
    data = df_clean[target_col].tolist()

    if len(data) > 1:
        # TÍNH TOÁN THỐNG KÊ
        n, mean, std = len(data), np.mean(data), np.std(data, ddof=1)
        sturges_k = int(1 + 3.322 * math.log10(n))
        
        # Công thức Ca, Cp, Cpk
        u_spec = (usl + lsl) / 2
        t_spec = usl - lsl
        ca = (mean - u_spec) / (t_spec / 2) if t_spec != 0 else 0
        cp = t_spec / (6 * std) if std != 0 else 0
        cpk = cp * (1 - abs(ca))
        
        ucl, lcl = mean + (sigma_val * std), mean - (sigma_val * std)
        
        # Tự động scale trục Y để không bị mất tiêu đề
        plot_min = min(lsl, lcl, min(data)) - 2
        plot_max = max(usl, ucl, max(data)) + 5 # Thêm 5 đơn vị để chừa chỗ cho tiêu đề
        
        config_dl = {'toImageButtonOptions': {'format': 'png', 'scale': 3}}

        # --- GIAO DIỆN CHÍNH ---
        st.markdown(f'<div class="pbi-header"><span style="font-size: 20px; font-weight: 700;">品質分析報告 | QC Analysis ({target_col})</span></div>', unsafe_allow_html=True)
        
        # KPI CARDS
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        metrics = [("樣本數 (N)", n), ("平均值 μ", f"{mean:.2f}"), ("標準差 σ", f"{std:.2f}"), 
                   ("Ca (準確度)", f"{ca:.2f}"), ("Cp (精密度)", f"{cp:.2f}"), ("Cpk (能力)", f"{cpk:.2f}")]
        cols = [k1, k2, k3, k4, k5, k6]
        for i, (label, val) in enumerate(metrics):
            cols[i].markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

        st.write("")

        # --- BIỂU ĐỒ SONG SONG (Parallel) ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            counts, bins = np.histogram(data, bins=sturges_k)
            bin_centers, bin_width = 0.5 * (bins[:-1] + bins[1:]), bins[1] - bins[0]
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, marker_color='#0078D4', opacity=0.7))
            x_curve = np.linspace(plot_min, plot_max, 500)
            y_curve = stats.norm.pdf(x_curve, mean, std) * n * bin_width
            fig_hist.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color='red', width=2)))
            
            fig_hist.add_vline(x=usl, line_dash="dash", line_color="#D83B01", annotation_text="USL")
            fig_hist.add_vline(x=lsl, line_dash="dash", line_color="#D83B01", annotation_text="LSL")

            fig_hist.update_layout(
                height=350, margin=dict(l=10, r=10, t=70, b=10), # t=70 là để chừa khoảng trống tiêu đề
                template="plotly_white",
                title=dict(text="數據分佈與常態曲線", font=dict(size=16), y=0.95, x=0.5, xanchor='center'),
                xaxis=dict(range=[plot_min, plot_max], title=y_label, mirror=True, showline=True, linecolor='black'),
                yaxis=dict(title="頻率", mirror=True, showline=True, linecolor='black'), showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True, config=config_dl)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=list(range(1, n+1)), y=data, mode='lines+markers', marker=dict(color='#0078D4', size=6)))
            
            fig_trend.add_hline(y=usl, line_dash="dash", line_color="#D83B01", annotation_text="USL")
            fig_trend.add_hline(y=lsl, line_dash="dash", line_color="#D83B01", annotation_text="LSL")
            fig_trend.add_hline(y=ucl, line_dash="dot", line_color="#107C10", annotation_text="UCL")
            fig_trend.add_hline(y=lcl, line_dash="dot", line_color="#107C10", annotation_text="LCL")

            fig_trend.update_layout(
                height=350, margin=dict(l=40, r=40, t=70, b=40),
                template="plotly_white",
                title=dict(text="趨勢監控與控制界限", font=dict(size=16), y=0.95, x=0.5, xanchor='center'),
                xaxis=dict(title=custom_x_label, mirror=True, showline=True, linecolor='black'),
                yaxis=dict(title=y_label, mirror=True, showline=True, linecolor='black', range=[plot_min, plot_max]),
                showlegend=False
            )
            st.plotly_chart(fig_trend, use_container_width=True, config=config_dl)
            st.markdown('</div>', unsafe_allow_html=True)

        # CHI TIẾT DỮ LIỆU
        st.markdown('<h3 style="color: #004E8C;">📋 詳細數據紀錄</h3>', unsafe_allow_html=True)
        df_clean['狀態'] = df_clean[target_col].apply(lambda x: '❌ OUT' if (x < lsl or x > usl) else '✅ PASS')
        st.dataframe(df_clean, use_container_width=True, hide_index=True)
