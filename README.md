# 📊 Stock Price Prediction App

This is a **Stock Price Prediction** web application built with **Streamlit**, **TensorFlow (LSTM)**, and **Yahoo Finance API**. The app fetches real-time stock data, displays company information, and predicts future stock prices using an LSTM neural network.

---

## 🚀 Features

- **📈 Real-time Stock Data**: Fetches stock price data from Yahoo Finance.
- **🏢 Company Information**: Displays company details, including CEO, sector, industry, and website.
- **🔮 Stock Price Prediction**: Uses an LSTM model to predict future stock prices.
- **📊 Interactive Charts**: Plot actual vs. predicted stock prices using Plotly.
- **🔍 Custom Ticker Search**: Select from popular stocks or enter a custom ticker.
- **⚡ Performance Metrics**: Evaluates model accuracy with MAE, MSE, and RMSE.

---

## 📂 Project Structure

```
📦 stock-price-prediction
├── 📄 app.py              # Main Streamlit app file
├── 📄 requirements.txt    # Dependencies
├── 📄 README.md           # Project Documentation
└── 📁 models              # Pre-trained model (optional)
```

---

## 🛠️ Installation

### 1️⃣ Clone the repository:
```sh
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2️⃣ Install dependencies:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the application:
```sh
streamlit run app.py
```

---

## 📌 Usage

1. Select a stock from the dropdown or enter a custom ticker.
2. Choose the date range for historical stock data.
3. Click the **Predict Stock Price** button.
4. View real-time stock data, company information, and predicted prices.

---

## 🏗️ Future Enhancements

- ✅ Save trained models to avoid retraining.
- ✅ Allow users to predict future stock prices beyond historical data.
- ✅ Optimize training with GPU acceleration.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Contributions

Contributions are welcome! Feel free to fork the repo and submit a pull request. 😊

