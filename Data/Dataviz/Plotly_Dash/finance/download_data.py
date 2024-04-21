import plotly.express as px
import yfinance as yf

df = yf.download("TSLA")
df.to_csv("TSLA.csv")

df = yf.download("MSFT")
df.to_csv("MSFT.csv")

df = yf.download("AAPL")
df.to_csv("AAPL.csv")


df = px.data.stocks()
df.to_csv("STOCKS.csv")
