import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Process Capability Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. CUSTOM CSS FOR PROFESSIONAL UI
st.markdown("""
    <style>
    /* Nền chính của ứng dụng */
    .stApp {
        background-color: #f4f7f6;
    }
    /* Làm đẹp các thẻ Metric */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 6px solid #008080;
    }
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #eef2f3;
    }
    /* Header trang */
    .main-header {
        background-color: #008080;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. GOOGLE SHEETS CONNECTION
def load_data():
    if "connections" in st.secrets and "gsheets" in st.secrets["connections"]:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            # ttl="1m" để tự động cập nhật dữ liệu mới sau mỗi 1 phút
            df = conn.read(spreadsheet=st.secrets["connections"]["gsheets"]["spreadsheet"], ttl="1m")
            return df
        except Exception as e:
            st.error(f"Error connecting to Google Sheets: {e}")
            return None
    else:
        st.error("Missing Google Sheets Configuration in Secrets!")
        return None

# MAIN APP LOGIC
df = load_data()

if df is not None:
    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("🛠️ Settings")
    
    # Chọn cột dữ liệu
    target_col = st.sidebar.selectbox("Select Data Column", df.columns)
    
    # Thiết lập USL/LSL
    usl = st.sidebar.number_input("Upper Spec Limit (USL)", value=0.1930, format="%.4f")
    lsl = st.sidebar.number_input("Lower Spec Limit (LSL)", value=0.1530, format="%.4f")
    
    # Xử lý dữ liệu số
    clean_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
    data_list = clean_data.tolist()

    if len(data_list) > 1:
        # --- CALCULATIONS ---
        mean = np.mean(data_list)
        std = np.std(data_list, ddof=1)
        max_v = np.max(data_list)
        min_v = np.min(data_list)
        n_samples = len(data_list)

        # Cp, Cpk, Ca
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpu = (usl - mean) / (3 * std) if std != 0 else 0
        cpl = (mean - lsl) / (3 * std) if std != 0 else 0
        cpk = min(cpu, cpl)
        ca = (mean - (usl + lsl)/2) / ((usl - lsl)/2)

        # --- UI DISPLAY ---
        st.markdown('<div class="main-header"><h1>Process Capability Analysis (Cp, Cpk)</h1></div>', unsafe_allow_html=True)

        # Row 1: Basic Statistics
        st.subheader("📋 Analysis Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sample Size (N)", n_samples)
        m2.metric("Mean", f"{mean:.4f}")
        m3.metric("StdDev", f"{std:.4f}")
        m4.metric("Max", f"{max_v:.4f}")
        m5.metric("Min", f"{min_v:.4f}")

        st.write("---")

        # Row 2: Capability Indices
        c_a, c_p, c_pk = st.columns(3)
        c_a.metric("Ca (Bias)", f"{ca:.2f}")
        c_p.metric("Cp (Precision)", f"{cp:.2f}")
        
        # Color-coded Cpk (Red if < 1.0, Orange < 1.33, Green >= 1.33)
        cpk_color = "#e74c3c" if cpk < 1.0 else "#f39c12" if cpk < 1.33 else "#27ae60"
        with c_pk:
            st.markdown(f"""
                <div style="background-color:{cpk_color}; padding:15px; border-radius:12px; text-align:center; color:white;">
                    <p style="margin:0; font-weight:bold; font-size:18px;">Cpk (Capability Index)</p>
                    <h1 style="margin:0; font-size:48px;">{cpk:.2f}</h1>
                </div>
                """, unsafe_allow_html=True)

        st.write("---")

       # Row 3: Histogram & Distribution
        st.subheader("📊 Measurement Histogram & Normal Distribution")
        
        # Calculate Bins to determine colors
        counts, bins = np.histogram(data_list, bins=12)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        # Determine colors: Red if outside Spec, Blue if inside
        bar_colors = ['#ff7f7f' if (x < lsl or x > usl) else '#5499c7' for x in bin_centers]

        fig = go.Figure()

        # 1. Histogram Bars (Frequency mode)
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_width * 0.9,
            marker_color=bar_colors,
            name='Actual Frequency',
            showlegend=False
        ))

        # 2. Normal Curve (Black, scaled to Frequency)
        x_curve = np.linspace(min(data_list + [lsl]) - 0.1, max(data_list + [usl]) + 0.1, 200)
        y_curve = stats.norm.pdf(x_curve, mean, std) * n_samples * bin_width
        
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve, 
            mode='lines', 
            name='Normal Curve', 
            line=dict(color='black', width=2)
        ))

        # 3. Stats Box (Top Left)
        stats_text = f"N = {n_samples}<br>Mean = {mean:.3f}<br>Std = {std:.3f}"
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text=stats_text, showarrow=False, align="left",
            bgcolor="white", bordercolor="black", borderwidth=1,
            font=dict(size=13)
        )

        # 4. LSL/USL Vertical Lines (Red Dashed)
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", line_width=2)
        fig.add_vline(x=usl, line_dash="dash", line_color="red", line_width=2)
        
        # Legend Item for Limits
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', 
                                 line=dict(color='red', dash='dash'), name='LSL / USL'))

        # 5. Layout Adjustments
        fig.update_layout(
            template="plotly_white",
            xaxis_title=f"{target_col} (LAB)",
            yaxis_title="Frequency",
            height=550,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.85, y=0.95, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No valid numeric data found in the selected column. Please check your Sheet.")

# 4. REQUIREMENTS.TXT CONTENT (Reminder)
# streamlit
# streamlit-gsheets-connection
# pandas
# numpy
# plotly
# scipy
