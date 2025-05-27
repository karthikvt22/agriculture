import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="🍅 Tomato AI Assistant", layout="centered")
st.markdown("<h1 style='text-align: center; color: tomato;'>🍅 Tomato AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Market Report (CSV or HTML)", type=["csv", "html"])

def parse_html_text(file):
    soup = BeautifulSoup(file.read(), "html.parser")
    text = soup.get_text()
    lines = text.splitlines()
    pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(\d+)\s+Tomato.*?(\d+)\s+(\d+)\s+(\d+)')
    records = []
    for line in lines:
        match = pattern.search(line)
        if match:
            date = datetime.strptime(match.group(1), "%d/%m/%Y")
            arrival = int(match.group(2))
            modal_price = int(match.group(5))
            records.append((date, arrival, modal_price))
    return pd.DataFrame(records, columns=["Date", "Arrival", "Price"])

def parse_csv(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Arrival", "Price"]]

if uploaded_file:
    # File parsing
    if uploaded_file.name.endswith(".html"):
        df = parse_html_text(uploaded_file)
    else:
        df = parse_csv(uploaded_file)

    df["Month"] = df["Date"].dt.month

    # Train model
    X = df[["Arrival", "Month"]]
    y = df["Price"]
    model = LinearRegression()
    model.fit(X, y)

    a, b = model.coef_
    c = model.intercept_

    st.success("✅ Model trained successfully")
    st.markdown(f"### 🧮 Equation:\n**Price = {a:.2f} × Arrival + {b:.2f} × Month + {c:.2f}**")

    # Prediction form
    st.markdown("---")
    st.subheader("🔍 Predict Tomato Price")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            arrival = st.slider("🚛 Arrival Volume (Tonnes)", 10, 300, 100)
        with col2:
            pred_date = st.date_input("📅 Market Date", date.today())
        submitted = st.form_submit_button("Predict")

    if submitted:
        month = pred_date.month
        predicted_price = model.predict([[arrival, month]])[0]
        st.success(f"💰 **Predicted Price**: ₹{int(predicted_price)} / quintal")

        # Plot: Arrival vs Price
        st.markdown("---")
        st.subheader("📊 Price vs Arrival Impact")
        arrivals = np.arange(10, 310, 10)
        prices = model.predict(np.column_stack((arrivals, np.full_like(arrivals, month))))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(arrivals, prices, color='tomato', marker='o')
        ax.set_xlabel("Arrival Volume (Tonnes)")
        ax.set_ylabel("Predicted Price (₹/quintal)")
        ax.set_title("Effect of Arrival on Tomato Price")
        st.pyplot(fig)
