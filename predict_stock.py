import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the saved model
model = load_model('nepse_prediction_model.h5')

# Load and preprocess the data
data = pd.read_csv('nepse.csv')  # Assumes your CSV file is named 'stock_data.csv'
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Create the input data for prediction
look_back = 5  # Adjust the look_back period according to your preference
last_week_data = scaled_prices[-30:]  # Get the last week's data
last_n_days = last_week_data[-look_back:]
X_test = np.reshape(last_n_days, (1, look_back, 1))

# Make predictions for the next week (assuming 7 days)
num_days = 7  # Number of days to predict
predicted_prices = []
current_data = X_test

for _ in range(num_days):
    predicted_price = model.predict(current_data)
    predicted_prices.append(predicted_price[0][0])
    current_data = np.append(current_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Determine buy/sell signals based on predicted prices
current_price = close_prices[-1]
signals = []

for price in predicted_prices:
    if price > current_price:
        signals.append("Buy")
    else:
        signals.append("Sell")
    current_price = price

# Create a DataFrame for the predicted prices and signals
predicted_data = pd.DataFrame({'Predicted Price': predicted_prices.flatten(), 'Signal': signals})

# Create a DataFrame for the last week's prices
last_week_prices = scaler.inverse_transform(last_week_data)
last_week_data = pd.DataFrame({'Last Week Price': last_week_prices.flatten()})

# Plot the last week's prices and predicted prices for the next week
plt.figure(figsize=(12, 6))
plt.plot(last_week_data.index, last_week_data['Last Week Price'], label='Last Week Price')
plt.plot(range(len(last_week_data), len(last_week_data) + len(predicted_data)), predicted_data['Predicted Price'], label='Predicted Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Last Week Prices and Predicted Prices for the Next Week')
plt.legend()
plt.show()

# Print the predicted prices and signals for the next week
print("Predicted Prices and Buy/Sell Signals for the Next Week:")
print(predicted_data)