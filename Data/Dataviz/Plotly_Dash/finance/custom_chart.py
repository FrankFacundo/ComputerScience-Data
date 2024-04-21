import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def custom_chart(df: pd.DataFrame):
    # Create a new figure
    fig = go.Figure()

    # Add FB line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["FB"],
            mode="lines",
            name="Facebook",
            line=dict(color="blue"),
        )
    )

    # Add AMZN line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["AMZN"],
            mode="lines",
            name="Amazon",
            line=dict(color="green"),
        )
    )

    # Add fill between the lines
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["FB"], mode="lines", name="Fill", fill=None)
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["AMZN"],
            mode="lines",
            fill="tonexty",  # Fill to next trace
            fillcolor="rgba(173,216,230,0.4)",  # Light blue fill with transparency
            line=dict(color="rgba(255,255,255,0)"),
        )
    )

    # Add FB line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["FB2"],
            mode="lines",
            name="Facebook2",
            line=dict(color="blue"),
        )
    )

    # Add AMZN line
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["AMZN2"],
            mode="lines",
            name="Amazon2",
            line=dict(color="green"),
        )
    )

    # Add fill between the lines
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["FB2"], mode="lines", name="Fill", fill=None)
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["AMZN2"],
            mode="lines",
            fill="tonexty",  # Fill to next trace
            fillcolor="rgba(173,216,230,0.4)",  # Light blue fill with transparency
            line=dict(color="rgba(255,255,255,0)"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Facebook vs Amazon Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        xaxis=dict(tickformat="%b\n%Y", dtick="M1"),
        hovermode="x",
    )

    # fig.show()
    return fig


if __name__ == "__main__":
    df = px.data.stocks()

    fig = custom_chart(df)
    fig.show()
