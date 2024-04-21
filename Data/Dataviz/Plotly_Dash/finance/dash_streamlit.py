import pathlib

import pandas as pd
import streamlit as st
from candlestick import get_candlestick_plot
from custom_chart import custom_chart

st.set_page_config(layout="wide")
# Sidebar options
ticker = st.sidebar.selectbox("Ticker to Plot", options=["TSLA", "MSFT", "AAPL"])

days_to_plot = st.sidebar.slider(
    "Days to Plot",
    min_value=1,
    max_value=300,
    value=120,
)
ma1 = st.sidebar.number_input(
    "Moving Average #1 Length",
    value=10,
    min_value=1,
    max_value=120,
    step=1,
)
ma2 = st.sidebar.number_input(
    "Moving Average #2 Length",
    value=20,
    min_value=1,
    max_value=120,
    step=1,
)

# Get the dataframe and add the moving averages
df = pd.read_csv(f"{pathlib.Path(__file__).parent.resolve()}/{ticker}.csv")
df[f"{ma1}_ma"] = df["Close"].rolling(ma1).mean()
df[f"{ma2}_ma"] = df["Close"].rolling(ma2).mean()
df = df[-days_to_plot:]

# Display the plotly chart on the dashboard
st.plotly_chart(
    get_candlestick_plot(df, ma1, ma2, ticker),
    use_container_width=True,
)

# stocks = px.data.stocks()
stocks = pd.read_csv(f"{pathlib.Path(__file__).parent.resolve()}/STOCKS.csv")

stocks["FB2"] = stocks["FB"] + 3
stocks["AMZN2"] = stocks["AMZN"] + 3

st.plotly_chart(
    custom_chart(stocks),
    use_container_width=True,
)
