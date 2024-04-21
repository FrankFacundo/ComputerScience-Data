"""
WARNING: dash_alternative_viz is a proof-of-concept and it is not in pypi. So not prod-ready. Use javascript instead.
"""

import random

import dash
import dash_alternative_viz as dav
from dash import html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.Button(id="my_button", children="More data!"),
        dav.HighChart(id="my_highchart"),
    ]
)


@app.callback(Output("my_highchart", "options"), [Input("my_button", "n_clicks")])
def random_chart(n_clicks):
    return {
        "title": {"text": "My heloooo"},
        "series": [
            {"data": [random.randrange(1, 50, 1) for i in range((n_clicks or 0) + 4)]}
        ],
    }


if __name__ == "__main__":
    app.run_server(debug=True)
