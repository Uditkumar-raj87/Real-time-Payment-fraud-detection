"""Streamlit risk operations dashboard."""

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fraud Risk Ops", layout="wide")
st.title("Fraud Risk Operations")
st.caption("Synthetic PaySim monitoring workspace")
api_url = os.getenv("FRAUD_API_URL", "http://localhost:8000")

uploaded = st.file_uploader("Upload a CSV of transactions", type=["csv"])
if uploaded is not None:
    frame = pd.read_csv(uploaded)
    st.metric("Transactions", len(frame))
    if "isFraud" in frame:
        st.metric("Observed fraud rate", f"{frame.isFraud.mean():.2%}")
    st.dataframe(frame.head(100), use_container_width=True)
    st.subheader("Investigator queue")
    st.dataframe(frame.sort_values("amount", ascending=False).head(20), use_container_width=True)
else:
    st.info("Upload a cleaned PaySim CSV to populate the investigator queue.")

st.sidebar.header("Score a transaction")
with st.sidebar.form("score"):
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    source = st.text_input("Origin account", "C123")
    destination = st.text_input("Destination account", "M456")
    submitted = st.form_submit_button("Score")
if submitted:
    payload = {"step": 1, "type": "TRANSFER", "amount": amount, "nameOrig": source, "oldbalanceOrg": amount, "newbalanceOrig": 0, "nameDest": destination, "oldbalanceDest": 0, "newbalanceDest": amount, "isFlaggedFraud": 0}
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=3)
        st.json(response.json())
    except requests.RequestException as error:
        st.error(f"API unavailable: {error}")