import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Fetch historical stock data
def fetch_stock_data(ticker, start="2020-01-01", end=None):
    print(f"Fetching stock data for {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']]

# Prepare data for LSTM
def prepare_lstm_data(data, time_steps=10):
    """Prepare data for LSTM (X, y) pairs."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Train LSTM model
def train_lstm_model(ticker="AMZN", time_steps=10):
    # Fetch and scale data
    data = fetch_stock_data(ticker)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data
    X, y = prepare_lstm_data(scaled_data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train model
    print("Training LSTM model...")
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Save model and scaler
    model.save(f"{ticker}_lstm_model.h5")
    joblib.dump(scaler, f"{ticker}_scaler.pkl")
    print(f"LSTM model and scaler for {ticker} saved successfully!")

if __name__ == "__main__":
    train_lstm_model("AMZN")