import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Traffic Sentinel", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model('my_model.h5')
        scaler = joblib.load('scaler.pkl')
        class_names = np.load('classes.npy', allow_pickle=True)
        return model, scaler, class_names
    except Exception as e:
        return None, None, None

model, scaler, class_names = load_assets()

# --- UI ---
st.title("🛡️ Encrypted Traffic Analysis")

if model is None:
    st.error("❌ System files missing. Please run the training script first!")
    st.stop()

# --- INPUT SECTION ---
st.subheader("1. Input Traffic Data")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

# Logic: Use Upload OR Demo
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File Loaded")
elif os.path.exists("demo_traffic.csv"):
    st.info("Using 'demo_traffic.csv' for demonstration.")
    df = pd.read_csv("demo_traffic.csv")

# --- ANALYSIS SECTION ---
if df is not None:
    # Clean data just in case
    if "Attack Type" in df.columns:
        X_input = df.drop("Attack Type", axis=1)
    else:
        X_input = df
        
    st.write(f"**Data Ready:** {len(X_input)} packets to analyze.")

    # THE BUTTON
    if st.button("🔴 Run Analysis"):
        status = st.empty()
        status.write("⏳ Normalizing data...")
        
        try:
            # 1. Scale
            X_scaled = scaler.transform(X_input)
            
            # 2. Predict
            status.write("🧠 Neural Network thinking...")
            preds = model.predict(X_scaled)
            
            # 3. Decode
            pred_classes = np.argmax(preds, axis=1)
            pred_names = [class_names[i] for i in pred_classes]
            
            status.write("✅ Analysis Complete!")
            
            # 4. Display Results (Simplified for speed)
            results = df.copy()
            results['Prediction'] = pred_names
            results['Confidence'] = np.max(preds, axis=1)
            
            # METRICS
            st.divider()
            n_malicious = np.sum(results['Prediction'] != 'Normal Traffic')
            
            c1, c2 = st.columns(2)
            c1.metric("Normal Flows", len(results) - n_malicious)
            c2.metric("Malicious Flows", n_malicious, delta_color="inverse")
            
            # CHART
            st.subheader("Threat Distribution")
            st.bar_chart(results['Prediction'].value_counts())
            
            # DATA
            st.subheader("Packet Logs")
            st.dataframe(results.head(10)) # Only show top 10 to save bandwidth
            
        except Exception as e:
            st.error(f"Error: {e}")
