# 🖥️ 9. Streamlit UI Example
# In app.py for Streamlit

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# Load dataset
df = pd.read_csv("AAPL.csv")  # Make sure AAPL.csv is in the same folder
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Close']]

st.title("📈 Stock Market Forecasting App")
arima_forecast = 129.25
lstm_forecast = 131.75
prophet_forecast = 130.60
ensemble = (arima_forecast + lstm_forecast + prophet_forecast) / 3
st.write(f"Next Forecasted Price: ₹{round(ensemble,2)}")

from statsmodels.tsa.arima.model import ARIMA

# Train ARIMA on historical Close prices
model_arima = ARIMA(df['Close'], order=(5, 1, 0))
result_arima = model_arima.fit()

# Forecast next steps (e.g., 30 future points)
forecast_arima = result_arima.forecast(steps=30)

# Ensure you provide x-values for forecast too
future_dates = pd.date_range(start=df.index[-1], periods=30, freq='B')

fig = go.Figure()
fig.add_trace(go.Scatter(y=df['Close'], mode='lines', name='Historical'))
fig.add_trace(go.Scatter(y=forecast_arima, mode='lines', name='ARIMA Forecast'))
st.plotly_chart(fig)