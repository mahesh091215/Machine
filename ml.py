import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="Stock Price Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Function for Prediction Logic ---
def run_prediction_model(ticker, prediction_days):
    """
    Downloads data, trains a Linear Regression model, and makes predictions.
    """
    st.subheader(f"📈 Data for {ticker}")

    # Define date range: last year of data up to today
    today = date.today()
    # Get enough data for training (2 years + 60 days buffer)
    start_date = today - timedelta(days=365*2 + 60) 

    # 1. Data Download
    @st.cache_data
    def load_data(ticker, start, end):
        try:
            # yfinance requires a string format for end date
            data = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
            data.reset_index(inplace=True)
            return data[['Date', 'Close']]
        except Exception as e:
            st.error(f"Error downloading data for {ticker}. Please ensure the ticker is correct and try again.")
            st.error(f"Details: {e}")
            return None

    data = load_data(ticker, start_date, today)

    if data is None or data.empty:
        st.warning("Could not load sufficient data or the ticker is invalid.")
        return

    st.write(data.tail())

    # 2. Feature Engineering
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    data['Lag_3'] = data['Close'].shift(3)
    # Drop the first 3 rows which contain NaN due to lagging
    data.dropna(inplace=True)

    # 3. Define Features (X) and Target (y)
    X = data[['Lag_1', 'Lag_2', 'Lag_3']]
    y = data['Close']

    # 4. Train-Test Split 
    test_size = prediction_days
    
    if len(X) < test_size:
        # Fallback if there isn't enough data for the requested test size
        st.warning(f"Not enough data ({len(X)} points) for a test set of size {test_size}. Reducing test size to 20.")
        test_size = 20
        
    # Split the data: Use the last 'test_size' rows for testing
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]


    # 5. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Model Prediction
    y_pred = model.predict(X_test)

    # 7. Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("⚙️ Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Mean Absolute Error (MAE)", value=f"${mae:.2f}")
    with col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"${rmse:.2f}")

    st.info(f"The model was tested on the last **{len(y_test)}** trading days of data.")

    # 8. Visualization
    st.subheader(f"📊 {ticker} Stock Price: Actual vs. Predicted")
    fig = plt.figure(figsize=(12, 6))
    
    # Get the dates corresponding to the test set
    test_dates = data['Date'].iloc[-len(y_test):].tolist()
    
    plt.plot(test_dates, y_test, label="Actual Prices", color="blue")
    plt.plot(test_dates, y_pred, label="Predicted Prices", color="red", linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"{ticker} Stock Price Prediction (Last {len(y_test)} Days)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    
    # 9. Future Prediction (The fixed section)
    st.subheader("🚀 Next Day Price Prediction")
    
    # Use the last three actual closing prices (Lag_1, Lag_2, Lag_3) to predict the next day
    last_known_data = X.iloc[-1].values.reshape(1, -1)
    
    # FIX: Use .item() to extract the single float value from the numpy array
    next_price_pred = model.predict(last_known_data).item()
    
    # Get the last date in the dataset and calculate the next trading day
    last_date = data['Date'].iloc[-1].date()
    # Note: This simply adds 1 day, which might not be a trading day, but it's illustrative.
    next_date = last_date + timedelta(days=1)
    
    st.markdown(f"The predicted closing price for the next day, **{next_date.strftime('%Y-%m-%d')}**, is: **${next_price_pred:.2f}**")
    st.markdown("*(Prediction based on the model's coefficients applied to the last three closing prices.)*")

# --- Streamlit UI Components ---
st.title("Stock Market Price Predictor (Linear Regression) 📈")
st.markdown("Predicting stock prices based on the previous three closing prices (`Lag_1`, `Lag_2`, `Lag_3`).")

# --- Sidebar for User Input ---
st.sidebar.header("User Input Parameters")

# Ticker Input
default_ticker = "AAPL"
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", default_ticker).upper()

# Prediction Days Slider
prediction_days = st.sidebar.slider(
    "Number of days to test/visualize prediction:",
    min_value=10,
    max_value=100,
    value=30,
    step=5
)

# Run Button
if st.sidebar.button("Run Prediction"):
    if ticker_input:
        run_prediction_model(ticker_input, prediction_days)
    else:
        st.sidebar.error("Please enter a stock ticker.")

st.sidebar.markdown("""
---
**Model Details:**
* **Model:** Linear Regression
* **Features:** Lag_1, Lag_2, Lag_3 
    (Closing prices from the previous 3 days)
* **Data Source:** Yahoo Finance (YFinance)
""")