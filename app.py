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
    # Load model and scaler only once
    try:
        model = tf.keras.models.load_model('my_model.h5')
        scaler = joblib.load('scaler.pkl')
        class_names = np.load('classes.npy', allow_pickle=True)
        return model, scaler, class_names
    except:
        return None, None, None

model, scaler, class_names = load_assets()

st.title("🛡️ Encrypted Traffic Analysis")

if model is None:
    st.error("Model files not found. Run the training script first!")
    st.stop()

# --- INPUT SECTION ---
st.write("### 1. Upload Traffic Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

# Use demo file if nothing uploaded
if uploaded_file is None and os.path.exists("demo_traffic.csv"):
    st.info("No file uploaded. Using 'demo_traffic.csv' automatically.")
    df = pd.read_csv("demo_traffic.csv")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
else:
    df = None

# --- ANALYSIS SECTION ---
if df is not None:
    # Prepare input
    if "Attack Type" in df.columns:
        X_input = df.drop("Attack Type", axis=1)
    else:
        X_input = df

    st.write(f"Loaded {len(X_input)} packets.")

    if st.button("RUN ANALYSIS"):
        # NO progress bars, NO animations - just do the work instantly
        try:
            # 1. Scale
            X_scaled = scaler.transform(X_input)

            # 2. Predict
            preds = model.predict(X_scaled)
            pred_classes = np.argmax(preds, axis=1)
            pred_names = [class_names[i] for i in pred_classes]

            # 3. Results
            results = df.copy()
            results['Prediction'] = pred_names

            st.success("Done!")

            # Summary
            st.write("### Results Summary")
            st.bar_chart(results['Prediction'].value_counts())

            # Detailed View
            st.write("### Detailed Logs")
            st.dataframe(results.head(10))

        except Exception as e:
            st.error(f"Error: {e}")
