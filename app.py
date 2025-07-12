import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import os

# Set date range
start = '2010-01-01'
end = '2024-12-31'

st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Add error handling for data download
try:
    df = yf.download(user_input, start=start, end=end)

    # Check if data is empty
    if df.empty:
        st.error(f"No data found for ticker: {user_input}")
        st.stop()

    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        st.error("Close price data not available")
        st.stop()

except Exception as e:
    st.error(f"Error downloading data: {str(e)}")
    st.stop()

# Describing Data
st.subheader('Data from 2010 - 2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.title('Closing Price vs Time')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(df['Close'], 'b', label='Closing Price')
plt.title('Closing Price vs Time with 100MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100MA')
plt.plot(ma200, 'g', label='200MA')
plt.plot(df['Close'], 'b', label='Closing Price')
plt.title('Closing Price vs Time with Moving Averages')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Check if we have enough data
if len(df) < 300:
    st.error("Not enough data for prediction. Need at least 300 days of data.")
    st.stop()

# Data Split
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Initialize scaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Check if model file exists
model_path = 'keras_model.h5'
if not os.path.exists(model_path):
    st.error(
        f"Model file '{model_path}' not found. Please ensure the model file is in the same directory as this script.")
    st.stop()

# Load model with error handling
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Use the same scaler that was fitted on training data
input_data = scaler.transform(final_df)

# Prepare test sequences
x_test, y_test = [], []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

# Convert to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Check if we have test data
if len(x_test) == 0:
    st.error("Not enough test data for prediction.")
    st.stop()

# Make predictions
try:
    y_predicted = model.predict(x_test)
except Exception as e:
    st.error(f"Error making predictions: {str(e)}")
    st.stop()

# Inverse transform to get actual prices
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Final Graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.title('Stock Price Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Display some metrics
st.subheader('Prediction Metrics')
mse = np.mean((y_test - y_predicted.flatten()) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_predicted.flatten()))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MSE", f"{mse:.2f}")
with col2:
    st.metric("RMSE", f"{rmse:.2f}")
with col3:
    st.metric("MAE", f"{mae:.2f}")