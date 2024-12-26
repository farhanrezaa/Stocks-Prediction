from flask import Flask, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load LSTM model and scaler
def load_lstm_model(ticker):
    model = load_model(f"{ticker}_lstm_model.h5")
    scaler = joblib.load(f"{ticker}_scaler.pkl")
    return model, scaler

# Fetch latest stock data
def fetch_latest_stock_data(ticker, time_steps=10):
    print(f"Fetching latest stock data for {ticker}...")
    stock = yf.download(ticker, period="1mo", interval="1d")  # Fetch last 15 days
    if stock.empty or len(stock) < time_steps:
        raise ValueError(f"Not enough data for {ticker}")
    return stock['Close'].values[-time_steps:]

# Predict tomorrow's price
def predict_tomorrow(ticker):
    try:
        model, scaler = load_lstm_model(ticker)
        data = fetch_latest_stock_data(ticker)
        scaled_data = scaler.transform(data.reshape(-1, 1))

        # Prepare input for LSTM
        X_input = np.array(scaled_data).reshape(1, len(scaled_data), 1)

        # Predict and rescale
        predicted_price = model.predict(X_input)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        return predicted_price
    except Exception as e:
        return str(e)

@app.route("/predict/<ticker>", methods=["GET"])
def predict(ticker):
    """API endpoint to predict tomorrow's stock price."""
    try:
        predicted_price = predict_tomorrow(ticker)
        return ticker, predicted_price
    except Exception as e:
        return "error"

if __name__ == "__main__":
    app.run(debug=True)