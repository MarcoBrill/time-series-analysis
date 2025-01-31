import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load your time series data
# Replace 'your_data.csv' with your actual data file
# The data should have a datetime index and a single column of values
data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Plot the original data
def plot_series(data, title="Time Series Data"):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

plot_series(data, "Original Time Series Data")

# Decompose the time series into trend, seasonal, and residual components
def decompose_series(data, model='additive'):
    decomposition = seasonal_decompose(data, model=model)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return trend, seasonal, residual

trend, seasonal, residual = decompose_series(data)

# Check for stationarity using the Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")

check_stationarity(data)

# If the data is non-stationary, apply differencing
def make_stationary(series):
    series_diff = series.diff().dropna()
    plot_series(series_diff, "Differenced Time Series")
    check_stationarity(series_diff)
    return series_diff

data_diff = make_stationary(data)

# ARIMA Model for forecasting
def arima_forecast(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

arima_model = arima_forecast(data_diff)

# LSTM Model for forecasting
def create_lstm_dataset(series, look_back=1):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back), 0])
        y.append(series[i + look_back, 0])
    return np.array(X), np.array(y)

def lstm_forecast(series, look_back=3, epochs=50):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    # Prepare the dataset
    X, y = create_lstm_dataset(series_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=2)

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

lstm_predictions = lstm_forecast(data_diff)

# Evaluate the models
def evaluate_model(actual, predicted):
    rmse = sqrt(mean_squared_error(actual, predicted))
    print('RMSE: %.3f' % rmse)
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.legend(loc='best')
    plt.show()

# Evaluate ARIMA model
arima_predictions = arima_model.predict(start=1, end=len(data_diff)-1, typ='levels')
evaluate_model(data_diff.values[1:], arima_predictions)

# Evaluate LSTM model
evaluate_model(data_diff.values[3:], lstm_predictions)

# Save the models (optional)
# arima_model.save('arima_model.pkl')
# lstm_model.save('lstm_model.h5')
