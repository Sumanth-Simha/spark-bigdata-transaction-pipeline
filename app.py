import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import io

from src.models.neural_net import FraudMLP
from src.data_preprocessing import preprocess_data

st.set_page_config(page_title="Fraud Detection Inference", layout="wide")

@st.cache_resource
def load_model():
    # Number of numerical features based on the dataset excluding 'Class' and 'Hour'
    # There are 30 numerical features typically (V1-V28, Amount, Time)
    num_features_count = 30
    embedding_dims = [(24, 4)]

    model = FraudMLP(num_features_count, embedding_dims)
    # Load model checkpoint. In production, this path could be configurable or dynamic
    try:
        model.load_state_dict(torch.load("src/models/checkpoints/best_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Model checkpoint not found. Please train the model first.")
        return None

    model.eval()
    return model

def run_inference(df, model, threshold):
    # Preprocess data for inference
    # Note: df will not have 'Class' for pure inference usually, but our preprocessing handles it
    try:
        x_num, x_cat, _, _ = preprocess_data(df, is_training=False, scaler_path="src/models/scaler.pkl")
    except Exception as e:
        st.error(f"Error during preprocessing. Ensure 'src/models/scaler.pkl' exists. Details: {e}")
        return None

    x_num = torch.tensor(x_num, dtype=torch.float32)
    x_cat = torch.tensor(x_cat, dtype=torch.long)

    with torch.no_grad():
        outputs = model(x_num, x_cat)
        probs = torch.sigmoid(outputs).numpy().flatten()

    df_results = df.copy()
    df_results['Fraud_Probability'] = probs

    # Flag transactions based on threshold
    df_results['Is_Suspicious'] = df_results['Fraud_Probability'] >= threshold

    return df_results

st.title("Credit Card Fraud Detection - Batch Inference")
st.write("Upload a CSV file containing transaction data to detect potential fraud.")

uploaded_file = st.file_uploader("Upload Transactions CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} transactions.")

    st.subheader("Inference Settings")
    threshold = st.slider("Suspicious Probability Threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

    model = load_model()

    if model is not None and st.button("Run Inference"):
        with st.spinner("Running inference..."):
            results_df = run_inference(df, model, threshold)

        if results_df is not None:
            suspicious_df = results_df[results_df['Is_Suspicious']].copy()

            st.subheader(f"Results: {len(suspicious_df)} suspicious transactions found.")

            if not suspicious_df.empty:
                st.dataframe(suspicious_df.sort_values(by='Fraud_Probability', ascending=False))

                # Download button for suspicious transactions
                csv = suspicious_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Flagged Transactions as CSV",
                    data=csv,
                    file_name="flagged_transactions.csv",
                    mime="text/csv",
                )
            else:
                st.success("No suspicious transactions found at this threshold.")
