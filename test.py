import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from yahooquery import search, Ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Initial greeting message
st.title("📊 Stock Price Prediction")
st.subheader("How can I help you? Which stock chart do you want to see?")

# Dropdown list of popular stocks
POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "BAC", "WMT"]

@st.cache_data
def get_ticker_by_name(company_name):
    results = search(company_name)
    quotes = results.get("quotes", [])
    return quotes[0]['symbol'].upper() if quotes else None

@st.cache_data
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            return None
        logo_url = f"https://logo.clearbit.com/{info.get('website', '').replace('https://', '').replace('http://', '')}" if info.get("website") else None
        if logo_url:
            response = requests.head(logo_url)
            logo_url = logo_url if response.status_code == 200 else None
        return {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A"),
            "ceo": info.get("companyOfficers", [{}])[0].get("name", "N/A"),
            "website": info.get("website", "N/A"),
            "logo": logo_url
        }
    except Exception:
        return None

@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, timeout=30)
        if stock_data.empty:
            raise ValueError("No data found")
        return stock_data
    except Exception:
        return None

# Sidebar inputs
st.sidebar.header("🔍 Search for Stock Prediction")
ticker = st.sidebar.selectbox("Choose a stock", POPULAR_TICKERS)
custom_ticker = st.sidebar.text_input("Or enter a custom ticker symbol:")
ticker = custom_ticker.upper() if custom_ticker else ticker
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")
sequence_length = st.sidebar.slider("Sequence Length", min_value=30, max_value=100, value=60, step=5)

if start_date >= end_date:
    st.sidebar.error("❌ Start date must be before end date!")

if st.sidebar.button("Predict Stock Price"):
    with st.spinner("Fetching company data..."):
        company_info = get_company_info(ticker)
    
    if not company_info:
        st.error("❌ Failed to fetch company information. Please try another ticker.")
    else:
        with st.spinner("Downloading stock data..."):
            stock_data = get_stock_data(ticker, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            st.sidebar.error("❌ No data found. Check ticker or date range.")
        else:
            st.subheader(f"📊 {company_info['name']} Stock Analysis")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if company_info["logo"]:
                    st.image(company_info["logo"], width=150, caption="Company Logo")
                
                # Check if CEO name is available
                ceo_name = company_info["ceo"]
                if ceo_name and ceo_name != "N/A":
                    ceo_photo_url = f"https://ui-avatars.com/api/?name={ceo_name.replace(' ', '+')}&size=150"
                    st.image(ceo_photo_url, width=150, caption=f"CEO: {ceo_name}")
                else:
                    st.warning("CEO information is not available.")
            with col2:
                st.markdown(f"**Sector:** {company_info['sector']}")
                st.markdown(f"**Industry:** {company_info['industry']}")
                st.markdown(f"**CEO:** {company_info['ceo']}")
                st.markdown(f"**[Website]({company_info['website']})**")
                st.markdown(company_info['summary'])
            
            tabs = st.tabs(["🔮 Predictions"])
            
            with tabs[0]:
                st.subheader("📈 Predicted vs Actual Stock Prices")
                data = stock_data[['Close']]
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)
                train_size = int(len(scaled_data) * 0.8)
                train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
                
                def create_sequences(data, sequence_length):
                    X, y = [], []
                    for i in range(sequence_length, len(data)):
                        X.append(data[i-sequence_length:i, 0])
                        y.append(data[i, 0])
                    return np.array(X), np.array(y)
                
                if len(train_data) > sequence_length and len(test_data) > sequence_length:
                    X_train, y_train = create_sequences(train_data, sequence_length)
                    X_test, y_test = create_sequences(test_data, sequence_length)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    # Define and train the model
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
                    st.info("Training the model... This may take a while.")
                    model.fit(X_train, y_train, batch_size=32, epochs=20, callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    predictions = scaler.inverse_transform(predictions)
                    true_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
                    test_dates = stock_data.index[-len(predictions):]
                    
                    # Plot predictions
                    pred_fig = go.Figure()
                    pred_fig.add_trace(go.Scatter(x=test_dates, y=true_prices.flatten(), mode='lines', name='Actual Prices', line=dict(color='blue')))
                    pred_fig.add_trace(go.Scatter(x=test_dates, y=predictions.flatten(), mode='lines', name='Predicted Prices', line=dict(color='red', dash='dot')))
                    pred_fig.update_layout(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
                    
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Display metrics
                    mae = mean_absolute_error(true_prices, predictions)
                    mse = mean_squared_error(true_prices, predictions)
                    rmse = np.sqrt(mse)
                    st.success(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
                    st.success(f"✅ Mean Squared Error (MSE): {mse:.2f}")
                    st.success(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")