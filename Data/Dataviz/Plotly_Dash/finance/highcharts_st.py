# import requests
import json
import pathlib

with open(f"{pathlib.Path(__file__).parent.resolve()}/aapl-ohlcv.json") as user_file:
    stock_data = json.load(user_file)

# stock_response = requests.get("https://demo-live-data.highcharts.com/aapl-ohlcv.json")
# stock_data = stock_response.text # This does not work

# To put many charts at the same time check: https://www.highcharts.com/blog/news/highcharts-version-9-2/

# candlestick_highcharts = {
#     "title": {
#         "text": "Candlestick and Heiken Ashi series comparison.",
#         "align": "left",
#     },
#     "rangeSelector": {"selected": 1},
#     "yAxis": [
#         {"title": {"text": "Candlestick"}, "height": "50%"},
#         {"title": {"text": "Heikin Ashi"}, "top": "50%", "height": "50%", "offset": 0},
#     ],
#     "series": [
#         {"type": "candlestick", "name": "Candlestick", "data": data},
#         {"type": "heikinashi", "name": "Heikin Ashi", "data": data, "yAxis": 1},
#     ],
# }

candlestick_highcharts = {
    "title": {"text": "AAPL Stock Price"},
    "range_selector": {"selected": 1},
    "navigator": {"enabled": True},
    "yAxis": [
        {"title": {"text": "Candlestick"}, "height": "100%"},
    ],
    "series": [{"type": "candlestick", "name": "AAPL Stock Price", "data": stock_data}],
}

# Need to add <script src="https://code.highcharts.com/stock/modules/hollowcandlestick.js"></script> into
# src/streamlit_highcharts/frontend/index.html from the repo "streamlit_highcharts"
candlestick_highcharts_2 = {
    "range_selector": {"selected": 2},
    "navigator": {"enabled": True, "series": {"color": "#2caffe"}},
    "series": [
        {"type": "hollowcandlestick", "name": "AAPL Stock Price", "data": stock_data}
    ],
}
