import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Process Capability Dashboard", layout="wide")

# Professional UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #f9f9f9; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border-left: 5px solid #008080;
    }
    .main-header {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-bottom: 3px solid #008080;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. DATA LOADING FROM GOOGLE SHEETS
def get_data():
    if "connections" in st.secrets:
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            # Refresh data every 60 seconds (ttl=60)
            return conn.read(spreadsheet=url, ttl=60)
        except Exception as e:
            st.error(f"Connection Error: {e}")
            return None
    else:
        st.error("Google Sheets configuration not found in Secrets!")
        return None

df = get_data()

if df is not None:
    # --- SIDEBAR: SETTINGS ---
    st.sidebar.header("📊 Configuration")
    target_col = st.sidebar.selectbox("Select Data Column", df.columns)
    
    # Set USL/LSL (Default based on your sample)
    usl = st.sidebar.number_input("Upper Spec Limit (USL)", value=-0.100, format="%.3f")
    lsl = st.sidebar.number_input("Lower Spec Limit (LSL)", value=-0.500, format="%.3f")
    
    # Numeric Data Cleaning
    raw_values = pd.to_numeric(df[target_col], errors='coerce').dropna()
    data = raw_values.tolist()

    if len(data) > 1:
        # --- CALCULATIONS ---
        n_samples = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        cp = (usl - lsl) / (6 * std) if std != 0 else 0
        cpk = min((usl - mean)/(3*std), (mean - lsl)/(3*std)) if std != 0 else 0
        ca = (mean - (usl + lsl)/2) / ((usl - lsl)/2)

        # --- UI DISPLAY ---
        st.markdown(f'<div class="main-header"><h1>Analysis: {target_col} (LAB)</h1></div>', unsafe_allow_html=True)

        # Row 1: Capability Indices
        c1, c2, c3 = st.columns(3)
        c1.metric("Ca (Bias)", f"{ca:.2f}")
        c2.metric("Cp (Precision)", f"{cp:.2f}")
        
        # Cpk Color Coding
        cpk_color = "green" if cpk >= 1.33 else "orange" if cpk >= 1.0 else "red"
        c3.markdown(f"**Cpk (Capability Index)**")
        c3.markdown(f"<h1 style='color:{cpk_color}; margin-top:-15px;'>{cpk:.2f}</h1>", unsafe_allow_html=True)

        st.write("---")

        # --- ADVANCED PLOT (MATCHING YOUR IMAGE) ---
        counts, bins = np.histogram(data, bins=12)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_width = bins[1] - bins[0]

        # Bar colors: Red for Out-of-Spec, Blue for In-Spec
        bar_colors = ['#ff7f7f' if (x < lsl or x > usl) else '#5499c7' for x in bin_centers]

        fig = go.Figure()

        # 1. Histogram Bars
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=counts,
            width=bin_width * 0.9,
            marker_color=bar_colors,
            name='Actual Frequency',
            showlegend=False
        ))

        # 2. Normal Distribution Curve
        x_curve = np.linspace(min(data + [lsl]) - 0.2, max(data + [usl]) + 0.2, 200)
        y_curve = stats.norm.pdf(x_curve, mean, std) * n_samples * bin_width
        
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve, 
            mode='lines', 
            line=dict(color='black', width=2),
            name='Normal Curve'
        ))

        # 3. Stats Info Box (N, Mean, Std) - Top Left
        stats_box = f"N = {n_samples}<br>Mean = {mean:.3f}<br>Std = {std:.3f}"
        fig.add_annotation(
            x=0.02, y=0.95, xref="paper", yref="paper",
            text=stats_box, showarrow=False, align="left",
            bgcolor="white", bordercolor="black", borderwidth=1,
            font=dict(size=13, color="black")
        )

        # 4. Spec Limit Lines (Red Dashed)
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", line_width=2)
        fig.add_vline(x=usl, line_dash="dash", line_color="red", line_width=2)
        
        # Dummy trace for Legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', 
                                 line=dict(color='red', dash='dash'), name='LSL / USL'))

        # 5. Layout Fine-tuning
        fig.update_layout(
            title=dict(text=f"Data Distribution: {target_col}", x=0.5),
            template="plotly_white",
            xaxis_title="Measurement Value",
            yaxis_title="Frequency",
            height=550,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(x=0.85, y=0.95, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Insufficient data for analysis. Please check your Google Sheet.")

else:
    st.info("💡 Please verify your Google Sheets connection in Streamlit Cloud Secrets.")
