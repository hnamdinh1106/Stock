import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

# Streamlit app title
st.title("Stock Price Prediction with ARIMA")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):")

if ticker:
    try:
        # Download data for the last 5 years
        df = yf.download(ticker, period='5y')
        df = df.reset_index()

        # Visualize the closing prices
        st.subheader(f'{ticker} Closing Prices Trend (Last 5 Years)')
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close'])
        plt.title('Closing Prices')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.grid()
        st.pyplot(plt)

        # Stationarity test
        def adfuller_test(close):
            result = adfuller(close)
            labels = ['ADF Test Statistics', 'p-value', '#Lags Used', 'Number of Observations Used']
            for value, label in zip(result, labels):
                st.write(f"{label}: {value}")
            if result[1] <= 0.05:
                st.write("Strong evidence against the null hypothesis (H0), data is stationary.")
            else:
                st.write("Weak evidence against the null hypothesis (H0), data is non-stationary.")

        adfuller_test(df['Close'])

        # Differencing
        df['Close_shift'] = df['Close'].diff().fillna(0)

        # Test for stationarity again
        st.subheader("Differencing Results")
        adfuller_test(df['Close_shift'])

        # Visualize the differenced series
        st.subheader("AAPL First Difference")
        plt.figure(figsize=(16, 8))
        plt.plot(df['Date'], df['Close_shift'])
        plt.title("Differenced Close Price")
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Differenced Close Price', fontsize=18)
        plt.grid()
        st.pyplot(plt)

        # ACF and PACF plots
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(df['Close_shift'].iloc[1:], lags=40, ax=ax[0])
        ax[0].set_title('ACF Plot')
        plot_pacf(df['Close_shift'].iloc[1:], lags=40, ax=ax[1])
        ax[1].set_title('PACF Plot')
        st.subheader("ACF and PACF Plots")
        st.pyplot(fig)

        # Train-test split
        train_data, test_data = df[0:int(len(df) * 0.8)], df[int(len(df) * 0.8):]

        # Fit auto ARIMA model
        model = auto_arima(train_data['Close_shift'], seasonal=False, stepwise=True, trace=True)

        # Predictions
        history = [x for x in train_data['Close_shift'].values]
        n_periods = len(test_data)
        model_fit = model.fit(history)
        predictions = model_fit.predict(n_periods=n_periods)

        # Calculate and print error metrics
        mse = mean_squared_error(test_data['Close_shift'], predictions)
        mae = mean_absolute_error(test_data['Close_shift'], predictions)

        st.write(f'MSE: {mse:.3f}')
        st.write(f'RMSE: {math.sqrt(mse):.3f}')
        st.write(f'MAE: {mae:.3f}')

        # Visualization of predictions
        plt.figure(figsize=(12, 7))
        plt.plot(test_data['Date'], predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
        plt.plot(test_data['Date'], test_data['Close_shift'], color='red', label='Actual Price')
        plt.title('Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Forecasting the next 20 days
        forecast_steps = 20
        forecast_values = []
        history = [x for x in train_data['Close_shift'].values]
        model_fit = model.fit(history)
        forecast_values = model_fit.predict(n_periods=forecast_steps)

        # Create a date range for the future dates
        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

        # Prepare the forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Close': forecast_values
        })

        # Select the last 20 actual prices for visualization
        actual_last_20_days = test_data.iloc[-20:]

        # Visualization of actual vs predicted future prices
        plt.figure(figsize=(12, 7))
        plt.plot(actual_last_20_days['Date'], actual_last_20_days['Close_shift'], color='red', label='Actual Price', marker='o')
        plt.plot(forecast_df['Date'], forecast_df['Forecasted_Close'], color='black', linestyle='dotted', label='Forecasted Price (Next 20 Days)')
        plt.title('Actual Prices vs Forecasted Prices for Next 20 Days')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
