import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path
import yfinance as yf

try:
    import tensorflow as tf
except:
    tf = None

st.set_page_config(page_title="Stock Predictor Pro", layout="wide")

st.title("📈 Stock Price Prediction (LSTM Pro)")
st.write("Multi-stock + Live Data + Prediction + Accuracy")

# Sidebar
st.sidebar.header("Settings")

stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "MSFT", "TSLA"])
use_live = st.sidebar.checkbox("Use Live Data (Yahoo Finance)", value=True)

# Load model + scaler
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("model/stock_model.h5")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# Load Data
if use_live:
    df = yf.download(stock, start="2022-01-01", end="2025-01-01")
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

# Ensure required columns
if 'date' not in df.columns:
    st.error("CSV must contain 'date' column")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 📊 Candlestick Chart
st.subheader("📊 Candlestick Chart")
st.write(df[['date','open','high','low','close']].tail())

# Simple candlestick using matplotlib
fig, ax = plt.subplots()
ax.plot(df['date'], df['close'], label="Close Price")
ax.set_title(f"{stock} Price Trend")
ax.legend()
st.pyplot(fig)

# Prepare Data
data_close = df[['close']].values
scaled_data = scaler.transform(data_close)

timesteps = 60

x_test = []
for i in range(timesteps, len(scaled_data)):
    x_test.append(scaled_data[i-timesteps:i, 0])

x_test = np.array(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Prediction
if st.button("Predict"):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1,1))

    real_data = data_close[-len(predictions):]

    # 📈 Plot
    st.subheader("📈 Prediction vs Real")
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(real_data, label="Real", color="red")
    ax2.plot(predictions, label="Predicted", color="blue")
    ax2.legend()
    st.pyplot(fig2)

    # 📊 RMSE
    rmse = np.sqrt(mean_squared_error(real_data, predictions))
    st.metric("Model RMSE", f"{rmse:.2f}")

    # 🔮 Next Day Prediction
    last_60 = scaled_data[-60:]
    X_input = last_60.reshape(1, 60, 1)

    next_pred = model.predict(X_input)
    next_price = scaler.inverse_transform(next_pred.reshape(-1,1))

    st.metric("Next Day Prediction", f"${next_price[0][0]:.2f}")