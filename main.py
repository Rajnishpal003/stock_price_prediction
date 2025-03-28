import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf  # type: ignore
from yahooquery import search  # type: ignore
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping
import time
from datetime import datetime
import matplotlib.dates as mdates

# Function to validate date input
def get_valid_date(prompt):
    while True:
        date_str = input(prompt).strip()
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Please enter in YYYY-MM-DD format.")

# Function to search for ticker symbols by company name
def get_ticker_by_name(company_name):
    print(f"Searching for the ticker symbol of '{company_name}'...")
    results = search(company_name)
    quotes = results.get("quotes", [])
    if not quotes:
        print("No companies found with that name.")
        return None
    print("Found the following matches:")
    for i, quote in enumerate(quotes[:5], 1):
        print(f"{i}. {quote['shortname']} ({quote['symbol']})")
    choice = input("Enter the number of the correct match (or '0' to exit): ")
    if choice.isdigit() and 1 <= int(choice) <= len(quotes[:5]):
        return quotes[int(choice) - 1]['symbol'].upper()
    else:
        print("Invalid choice or exit selected.")
        return None

# Ask for company name or ticker symbol
choice = input("Do you know the ticker symbol? (yes/no): ").strip().lower()
if choice == "yes":
    ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ").strip().upper()
else:
    company_name = input("Enter the company name: ").strip()
    ticker = get_ticker_by_name(company_name)
    if not ticker:
        print("Exiting program as no ticker symbol was selected.")
        exit()

# Get date range from user
start_date = get_valid_date("Enter the start date (YYYY-MM-DD): ")
end_date = get_valid_date("Enter the end date (YYYY-MM-DD): ")
if start_date >= end_date:
    print("Start date must be before the end date.")
    exit()

# Fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if stock_data.empty:
                print("No data found. Check ticker or date range.")
                return None
            return stock_data
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

stock_data = fetch_stock_data(ticker, start_date, end_date)
if stock_data is None:
    exit()

data = stock_data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 60
if len(train_data) > sequence_length and len(test_data) > sequence_length:
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    print("Training the model...")
    model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[early_stop])
    
    print("Predicting stock prices...")
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_dates = stock_data.index[-len(true_prices):]

    results = pd.DataFrame({
        'Date': test_dates,
        'Actual Price': true_prices.flatten(),
        'Predicted Price': predictions.flatten()
    })
    print(results)
    results.to_csv(f"{ticker}_predictions.csv", index=False)
    print(f"Predictions saved to {ticker}_predictions.csv")

    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, true_prices, color='blue', label='Actual Prices')
    plt.plot(test_dates, predictions, color='red', label='Predicted Prices')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.show()
else:
    print(f"Not enough data to create sequences with a sequence length of {sequence_length} days.")
