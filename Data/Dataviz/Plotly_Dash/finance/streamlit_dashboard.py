import pathlib

import pandas as pd
import requests
import streamlit as st
import streamlit_highcharts as hg
from candlestick import get_candlestick_plot
from custom_chart import custom_chart
from highcharts_st import candlestick_highcharts, candlestick_highcharts_2
from highcharts_stock.chart import Chart
from streamlit_lightweight_charts import renderLightweightCharts
from trading_view_st import (
    chartMultipaneOptions,
    seriesCandlestickChart,
    seriesMACDchart,
    seriesVolumeChart,
)

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

st.subheader("Highcharts")
stock_response = requests.get("https://demo-live-data.highcharts.com/aapl-ohlcv.json")
stock_data = stock_response.text

as_dict = {
    "range_selector": {"selected": 2},
    "title": {"text": "AAPL Stock Price"},
    "navigator": {"enabled": True},
    "series": [{"type": "candlestick", "name": "AAPL Stock Price", "data": stock_data}],
    # "stockTools": {"gui": {"enabled": True}}, # https://www.highcharts.com/docs/stock/stock-tools
}

chart = Chart.from_options(as_dict)
html_object = chart._repr_html_()
st.components.v1.html(html_object, height=400)

st.subheader("Plotly custom chart")

# stocks = px.data.stocks()
stocks = pd.read_csv(f"{pathlib.Path(__file__).parent.resolve()}/STOCKS.csv")

stocks["FB2"] = stocks["FB"] + 3
stocks["AMZN2"] = stocks["AMZN"] + 3

st.plotly_chart(
    custom_chart(stocks),
    use_container_width=True,
)

st.subheader("Plotly candlestick")
# Display the plotly chart on the dashboard
st.plotly_chart(
    get_candlestick_plot(df, ma1, ma2, ticker),
    use_container_width=True,
)


# Adapted from: https://github.com/freyastreamlit/streamlit-lightweight-charts/tree/main
# TODO: Check https://stackoverflow.com/questions/76892626/price-scale-width-inconsistent-for-lightweight-chart-tradingview-with-multiple
st.subheader("LightweightCharts by Trading View")

renderLightweightCharts(
    [
        {"chart": chartMultipaneOptions[0], "series": seriesCandlestickChart},
        {"chart": chartMultipaneOptions[1], "series": seriesVolumeChart},
        {"chart": chartMultipaneOptions[2], "series": seriesMACDchart},
    ],
    "multipane",
)

st.subheader("Highcharts2")
# Adapted from:
# https://aalteirac-streamlit-highcharts-test-app-main-3vgde6.streamlit.app/
# https://www.highcharts.com/blog/tutorials/how-to-read-hollow-candlesticks/
# https://www.highcharts.com/samples/stock/demo/hollow-candlestick?jsfiddle
# https://www.highcharts.com/blog/news/highcharts-version-9-2/
hg.streamlit_highcharts(candlestick_highcharts_2)
st.subheader("Highcharts3")
hg.streamlit_highcharts(candlestick_highcharts)


# TODO: Add UMAP and t-SNE: https://plotly.com/python/t-sne-and-umap-projections/
