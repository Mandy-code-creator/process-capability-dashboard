import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Page Config
st.set_page_config(page_title="Process Capability Dashboard", layout="wide")

st.title("Process Capability Analysis (Cp, Cpk)")
st.markdown("---")

# --- 1. Connection & Data Loading ---
# Make sure your Google Sheet column header is "Measurements"
sheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(spreadsheet=sheet_url, ttl="1m")
    
    # Sidebar for Settings
    st.sidebar.header("Specifications")
    usl = st.sidebar.number_input("Upper Spec Limit (USL)", value=0.193, format="%.3f")
    lsl = st.sidebar.number_input("Lower Spec Limit (LSL)", value=0.153, format="%.3f")
    
    target_col = st.sidebar.selectbox("Select Data Column", df.columns)
    
    # Data Cleaning
    raw_data = pd.to_numeric(df[target_col], errors='coerce').dropna()
    data = raw_data.tolist()

    if len(data) > 0:
        # --- 2. Calculations ---
        mean = np.mean(data)
        std = np.std(data, ddof=1) # Sample standard deviation
        max_val = np.max(data)
        min_val = np.min(data)
        n = len(data)

        # Capability Indices
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpu = (usl - mean) / (3 * std) if std != 0 else 0
        cpl = (mean - lsl) / (3 * std) if std != 0 else 0
        cpk = min(cpu, cpl)
        ca = (mean - (usl + lsl)/2) / ((usl - lsl)/2)

        # --- 3. UI: Analysis Results ---
        st.subheader("Analysis Results")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sample Size (N)", n)
        m2.metric("Mean", f"{mean:.4f}")
        m3.metric("StdDev", f"{std:.4f}")
        m4.metric("Max", f"{max_val:.4f}")
        m5.metric("Min", f"{min_val:.4f}")

        st.markdown("### Capability Indices")
        i1, i2, i3 = st.columns(3)
        i1.metric("Ca (Bias)", f"{ca:.2f}")
        i2.metric("Cp (Precision)", f"{cp:.2f}")
        
        # Color-coded Cpk
        cpk_color = "green" if cpk >= 1.33 else "orange" if cpk >= 1.0 else "red"
        i3.markdown(f"**Cpk (Capability)**")
        i3.markdown(f"<h1 style='color: {cpk_color}; margin-top:-20px;'>{cpk:.2f}</h1>", unsafe_allow_html=True)

        # --- 4. UI: Histogram ---
        st.subheader("Measurement Histogram")
        
        # Normal Distribution Curve
        x_curve = np.linspace(min(data + [lsl]) * 0.98, max(data + [usl]) * 1.02, 200)
        y_curve = stats.norm.pdf(x_curve, mean, std)

        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=data, 
            nbinsx=15, 
            name='Actual Data',
            histnorm='probability density',
            marker_color='#45B39D'
        ))
        
        # Curve
        fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Normal Curve', line=dict(color='#2E86C1', width=3)))

        # Spec Limit Lines
        fig.add_vline(x=lsl, line_dash="dash", line_color="#C0392B", annotation_text="LSL")
        fig.add_vline(x=usl, line_dash="dash", line_color="#C0392B", annotation_text="USL")
        fig.add_vline(x=mean, line_color="#2ECC71", annotation_text="Mean")

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            height=500,
            xaxis_title="Measurement Value",
            yaxis_title="Density"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No valid numeric data found in the selected column.")

except Exception as e:
    st.error(f"Connection Error: {e}")
    st.info("Please check your Google Sheet link and 'Anyone with the link can view' settings.")
