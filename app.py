import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

try:
    import tensorflow as tf
    TENSORFLOW_IMPORT_ERROR = None
except Exception as exc:
    tf = None
    TENSORFLOW_IMPORT_ERROR = exc

# Page Configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Google Stock Price Prediction App")
st.write("Ye app LSTM model ka use karke Google ke stock prices predict karta hai.")

# Sidebar for controls
st.sidebar.header("User Input")

# 1. Load Model and Scaler
@st.cache_resource
def load_resources():
    model_path = Path("model/stock_model.h5")
    scaler_path = Path("model/scaler.pkl")

    if TENSORFLOW_IMPORT_ERROR is not None:
        raise RuntimeError(
            "TensorFlow is not installed in the active Python environment. "
            "Create a Python 3.10/3.11 virtualenv and install requirements first."
        ) from TENSORFLOW_IMPORT_ERROR

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Missing trained artifacts: expected model/stock_model.h5 and model/scaler.pkl"
        )

    model = tf.keras.models.load_model("model/stock_model.h5")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_resources()
    st.sidebar.success("Model & Scaler Loaded Successfully!")
except Exception as e:
    st.error(f"App startup failed. Error: {e}")
    st.stop()

# 2. Load Data (Display purpose ke liye)
uploaded_file = st.sidebar.file_uploader("Upload CSV File (GOOG.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing (Same as your training logic)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values(by='date')
    
    st.subheader("Raw Data Visualization")
    st.line_chart(df.set_index('date')['close'])

    # Data Preparation for Prediction
    data_close = df.filter(['close']).values
    
    # Scaling
    scaled_data = scaler.transform(data_close)
    
    # Create Test Data (Last 60 days logic)
    timesteps = 60
    
    # Hum maan lete hain user poora data de raha hai aur humein end ka predict karna hai
    # Yahan hum wahi logic laga rahe hain jo aapne test set ke liye use kiya
    if len(scaled_data) > timesteps:
        # Prepare Data for plotting predictions on existing data
        # (Simplify karke hum last 100 days ka graph dikhayenge)
        
        test_data = scaled_data[-(timesteps + 100):] 
        x_test = []
        # y_test actually future value hai, yahan hum bas demo ke liye x banayenge
        
        for i in range(timesteps, len(test_data)):
            x_test.append(test_data[i-timesteps:i, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Prediction
        if st.button("Predict Prices"):
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Real values for comparison (unscaled)
            real_data = data_close[-(len(predictions)):]
            
            # Plotting using Matplotlib inside Streamlit
            st.subheader("Prediction vs Original")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(real_data, color='red', label='Real Price')
            ax.plot(predictions, color='blue', label='Predicted Price')
            ax.set_title('Google Stock Price Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
            
            # Next Day Prediction Logic
            last_60_days = scaled_data[-60:]
            X_input = last_60_days.reshape(1, 60, 1)
            next_day_pred = model.predict(X_input)
            next_day_price = scaler.inverse_transform(next_day_pred)
            
            st.metric(label="Predicted Price for Next Day", value=f"$ {next_day_price[0][0]:.2f}")
            
    else:
        st.error(f"Data bohot chota hai. Kam se kam {timesteps} rows chahiye.")

else:
    st.info("Kripya CSV file upload karein prediction dekhne ke liye.")
