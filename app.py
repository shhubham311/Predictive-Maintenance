import streamlit as st
import requests
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import joblib

# Suppress specific warnings to keep UI clean
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pandas.*")

# Import SHAP
import shap

# Page Config
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Intelligent Predictive Maintenance Platform")
st.caption("Featuring Hybrid Detection (XGBoost + Isolation Forest) + SHAP Explainability")

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_training_dataset():
    """
    Loads real NASA dataset to pick random realistic samples.
    """
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    try:
        df = pd.read_csv('data/raw/train_FD001.txt', sep=' ', names=columns, index_col=False)
        # Clean potential empty columns from sep=' '
        df = df.replace('', np.nan).dropna(axis=1, how='all')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def get_healthy_sample():
    """Returns exact values from 'Healthy Engine' (Row 1)."""
    return {
        'setting1': -0.0007, 'setting2': -0.0004, 'setting3': 100.0,
        's1': 518.67, 's2': 641.82, 's3': 1589.7, 's4': 1400.6,
        's5': 14.62, 's6': 21.61, 's7': 554.36, 's8': 2388.06,
        's9': 9046.2, 's10': 1.3, 's11': 47.3, 's12': 521.66,
        's13': 2388.02, 's14': 8138.62, 's15': 8.3199,
        's16': 0.03, 's17': 391.0, 's18': 2388.0,
        's19': 100.0, 's20': 39.0, 's21': 23.4194
    }

def get_failing_sample():
    """Returns values from a 'Failing Engine' (High Temp/Vibration)."""
    return {
        'setting1': 0.0, 'setting2': 0.0, 'setting3': 100.0,
        's1': 600.0,  's2': 700.0,  's3': 1600.0, 's4': 1450.0,
        's5': 20.0,   's6': 25.0,   's7': 600.0,  's8': 2500.0,
        's9': 9500.0, 's10': 2.0,   's11': 55.0,   's12': 550.0,
        's13': 2450.0, 's14': 8300.0, 's15': 9.0,
        's16': 0.04,   's17': 400.0,  's18': 2400.0,
        's19': 110.0,  's20': 45.0,   's21': 25.0
    }

def get_random_sample_from_data():
    """Picks a random row from real NASA training dataset."""
    df = load_training_dataset()
    if df.empty:
        return get_healthy_sample()
    
    random_row = df.sample(1).iloc[0]
    sample = {}
    
    def add_val(key):
        if key in random_row and pd.notna(random_row[key]):
            val = random_row[key]
            noise = val * 0.005 * (np.random.rand() - 0.5)
            sample[key] = float(val + noise)
        else:
            sample[key] = 0.0

    add_val('setting1')
    add_val('setting2')
    add_val('setting3')
    for i in range(1, 22):
        add_val(f's{i}')
    return sample

# --- SESSION STATE INITIALIZATION ---
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = get_healthy_sample()

# --- LOAD ARTIFACTS ---
def load_artifacts():
    try:
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        pipeline = joblib.load('models/advanced_model.pkl')
        return metrics, pipeline
    except FileNotFoundError:
        return None, None

metrics, pipeline_obj = load_artifacts()

# --- SIDEBAR ---
st.sidebar.header("Configuration")
input_method = st.sidebar.radio("Input Mode", ["🎲 Smart Random (Auto-Fill)", "✏️ Manual Input"])

if input_method == "🎲 Smart Random (Auto-Fill)":
    st.sidebar.subheader("Quick Presets")
    if st.sidebar.button("✅ Generate Healthy", use_container_width=True):
        st.session_state.sensor_data = get_healthy_sample()
        st.rerun()
        
    if st.sidebar.button("🎲 Generate Random (Real Data)", type="secondary", use_container_width=True):
        st.session_state.sensor_data = get_random_sample_from_data()
        st.rerun()
        
    if st.sidebar.button("🔥 Simulate Failing", use_container_width=True):
        st.session_state.sensor_data = get_failing_sample()
        st.rerun()
        
else:
    st.sidebar.subheader("Manual Inputs")
    manual_data = {}
    cols = [f's{i}' for i in range(1, 22)] + ['setting1', 'setting2', 'setting3']
    for col in cols:
        val = st.sidebar.number_input(col, value=st.session_state.sensor_data.get(col, 500.0), key=col)
        manual_data[col] = val
    st.session_state.sensor_data = manual_data


# --- MAIN LAYOUT ---

# SECTION 1: LIVE PREDICTION
st.header("🔴 Real-Time Monitoring Dashboard")
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.subheader("1. Sensor Input Data")
    st.json(st.session_state.sensor_data)
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        try:
            response = requests.post("http://localhost:8000/predict", json=st.session_state.sensor_data)
            if response.status_code == 200:
                res = response.json()
                st.session_state['result'] = res
        except requests.exceptions.ConnectionError:
            st.error("❌ Backend API Offline. Run `uvicorn api:app --reload`")

with col2:
    st.subheader("2. Prediction Results")
    if 'result' in st.session_state:
        res = st.session_state['result']
        rul = res['predicted_rul']
        anomaly = res['anomaly']
        status = res['status']
        
        st.metric("Predicted RUL (Cycles)", f"{rul:.1f}")
        
        if anomaly:
            st.error(f"⚠️ {status}")
        elif status == "Critical":
            st.warning(f"⚠️ {status}")
        else:
            st.success(f"✅ {status}")
            
        prog = min(1.0, max(0.0, rul / 200))
        st.progress(prog)

with col3:
    st.subheader("3. System Health")
    if metrics:
        st.metric("Model RMSE", f"{metrics['RMSE']:.2f}")
        st.metric("Model R²", f"{metrics['R2_Score']:.2f}")
        st.caption(f"Contamination: 5%")

st.markdown("---")

# SECTION 2: ANALYTICS & EXPLAINABILITY
st.header("📊 Model Analytics & Explainability")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("1. Performance Metrics")
    if metrics:
        st.write(f"**Best Params:** {metrics['Best_Params']}")
    if os.path.exists('img/feature_importance.png'):
        st.image('img/feature_importance.png', caption="Feature Importance (Gain)")
    else:
        st.warning("Run training script to generate plots.")

with col_b:
    st.subheader("2. Residuals Distribution")
    if os.path.exists('img/residual_plot.png'):
        st.image('img/residual_plot.png', caption="Model Residuals")
    else:
        st.warning("Run training script to generate plots.")

# SECTION 3: SHAP EXPLAINER (Final Stable Version)
st.header("🔍 Explainability (SHAP)")

if st.button("Explain Last Prediction with SHAP"):
    if pipeline_obj:
        with st.spinner("Generating Explanation..."):
            try:
                # Load model and scaler
                model = pipeline_obj['model']
                scaler = pipeline_obj['scaler']
                training_cols = pipeline_obj['feature_cols']
                
                # Prepare input using SESSION STATE data
                df_input = pd.DataFrame([st.session_state.sensor_data])
                
                # Manual FE
                df_input['setting1_x_s1'] = df_input['setting1'] * df_input['s1']
                df_input['setting2_x_s2'] = df_input['setting2'] * df_input['s2']
                df_input['setting3_x_s3'] = df_input['setting3'] * df_input['s3']
                
                # Ensure columns match training data
                for col in training_cols:
                    if col not in df_input.columns:
                        df_input[col] = 0
                
                # --- TYPE CASTING ---
                # Keep DataFrame for Scaler
                X_input_df = df_input[training_cols].astype(float)
                
                # Scale
                X_input_scaled = scaler.transform(X_input_df).astype(float)
                
                # --- SHAP EXPLANATION ---
                explainer = shap.TreeExplainer(model)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_input_scaled)
                
                # --- FORCE PLOT (Standard Professional) ---
                # Now that XGBoost is 3.0.0, 'expected_value' is a float (not a string list).
                # This makes Force Plot fully stable again.
                st.subheader("Force Plot: Feature Impact")
                fig = plt.figure()
                
                shap.force_plot(
                    float(explainer.expected_value), # Explicit cast
                    shap_values[0], 
                    feature_names=list(training_cols),
                    matplotlib=True,
                    show=False # Prevents popup window
                )
                
                # Use plt.gcf() to grab the current figure safely
                st.pyplot(plt.gcf())
                
                st.caption("Red bars push RUL DOWN (Critical), Blue bars push RUL UP (Healthy).")
                
            except Exception as e:
                st.error(f"SHAP Error: {e}")
                st.caption("Ensure XGBoost is version 3.0.0 and backend is restarted.")
    else:
        st.error("Pipeline not loaded.")