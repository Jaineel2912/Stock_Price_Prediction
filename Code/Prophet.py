import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date

st.title('Stock Trend Prediction')

selected_stock = st.text_input('Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)')

start = '2000-01-01'
end = date.today()

# Fetch stock data using yfinance
try:
    stock_data = yf.download(selected_stock, start=start, end=end).reset_index()
    st.subheader(f'Data from {start} - {end}')
    st.write(stock_data.describe())

    # Forecasting
    st.title("Stock Prediction App")

    n_years = st.text_input("Years Of Prediction:")
    period = int(n_years) * 365

    df_train = stock_data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    st.subheader('All Forecasted Data')
    st.write(forecast)

    st.write('Forecast Data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Visualization
    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Close'])
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA')
    ma100 = stock_data['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(stock_data['Close'])
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = stock_data['Close'].rolling(100).mean()
    ma200 = stock_data['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100 ,label='100-day MA', linestyle='--')
    plt.plot(ma200, label='200-day MA', linestyle='--')
    plt.plot(stock_data['Close'],label='Actual Prices')
    plt.plot(forecast['yhat_upper'],label='Forecasted Prices', linestyle='-')
    plt.legend()
    st.pyplot(fig)

    # SIP Calculator with interest rate
    st.subheader('SIP Calculator')
    investment = st.number_input('Enter Monthly Investment Amount', value=1)
    years = st.number_input('Investment Period (in years)', value=1)
    annual_interest_rate = st.number_input('Annual Interest Rate (%)', value=1)

    total_investment = investment * 12 * years

    # Calculate the compounded growth using compound interest formula
    rate = annual_interest_rate / 100
    future_value = total_investment * ((1 + rate) ** years)

    projected_growth = forecast.loc[forecast['ds'] >= forecast['ds'].max() - pd.DateOffset(years=years)]
    initial_price = projected_growth.iloc[0]['yhat']
    final_price = projected_growth.iloc[-1]['yhat']

    profit_percentage = ((future_value - total_investment) / total_investment) * 100
    projected_percentage = ((final_price - initial_price) / initial_price) * 100

    st.write(f"Total Investment: ${total_investment}")
    st.write(f"Projected Growth with SIP: ${future_value:.2f}")
    st.write(f"Projected Growth from Stock: ${final_price:.2f}")
    st.write(f"SIP Projected Growth: {profit_percentage:.2f}%")
    st.write(f"Stock Projected Growth: {projected_percentage:.2f}%")

    # Pie Chart representing Investment and Profit Percentage
    fig, ax = plt.subplots()
    sizes = [total_investment, future_value]
    labels = ['Total Investment', 'Projected Growth']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')

    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
