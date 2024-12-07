import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

ticker = "AAPL"
start_date = "2023-01-01"
end_date = "2024-12-31"

data = yf.download(ticker, start=start_date, end=end_date)
data.reset_index(inplace=True)
data = data[['Date', 'Close']]

data['Lag_1'] = data['Close'].shift(1)
data['Lag_2'] = data['Close'].shift(2)
data['Lag_3'] = data['Close'].shift(3)
data.dropna(inplace=True)

X = data[['Lag_1', 'Lag_2', 'Lag_3']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

plt.figure(figsize=(14, 7))
plt.plot(data['Date'].iloc[-len(y_test):], y_test, label="Actual Prices", color="blue")
plt.plot(data['Date'].iloc[-len(y_test):], y_pred, label="Predicted Prices", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.legend()
plt.show()
