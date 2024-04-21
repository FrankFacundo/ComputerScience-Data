import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def get_candlestick_plot(df: pd.DataFrame, ma1: int, ma2: int, ticker: str):
    """
    Create the candlestick chart with two moving avgs + a plot of the volume
    Parameters
    ----------
    df : pd.DataFrame
        The price dataframe
    ma1 : int
        The length of the first moving average (days)
    ma2 : int
        The length of the second moving average (days)
    ticker : str
        The ticker we are plotting (for the title).
    """

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{ticker} Stock Price", "Volume Chart"),
        row_width=[0.3, 0.7],
    )

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick chart",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Line(x=df["Date"], y=df[f"{ma1}_ma"], name=f"{ma1} SMA"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Line(x=df["Date"], y=df[f"{ma2}_ma"], name=f"{ma2} SMA"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=df["Date"], y=df["Volume"], name="Volume"),
        row=2,
        col=1,
    )

    fig["layout"]["xaxis2"]["title"] = "Date"
    fig["layout"]["yaxis"]["title"] = "Price"
    fig["layout"]["yaxis2"]["title"] = "Volume"

    fig.update_xaxes(
        rangebreaks=[{"bounds": ["sat", "mon"]}],
        rangeslider_visible=False,
    )

    return fig


if __name__ == "__main__":
    # import yfinance as yf
    # df = yf.download("TSLA")
    # df.to_csv("TSLA.csv")
    df = pd.read_csv("TSLA.csv")

    df["10_ma"] = df["Close"].rolling(10).mean()
    df["20_ma"] = df["Close"].rolling(20).mean()

    fig = get_candlestick_plot(df[-120:], 10, 20, "TSLA")
    fig.show()
