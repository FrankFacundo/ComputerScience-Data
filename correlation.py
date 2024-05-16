import pandas as pd
import yfinance as yf


def compute_correlation(stock1, stock2):
    """
    Compute the correlation between two stocks using daily closing prices.
    Compute the correlation between two stocks using daily closing prices,
    and save the data for each stock to a CSV file.

    Args:
        stock1 (str): Ticker symbol for the first stock.
        stock2 (str): Ticker symbol for the second stock.

    Returns:
        float: The correlation coefficient between the two stocks.
    """
    # Fetch historical data for both stocks
    data1 = yf.download(stock1, start="2020-01-01", end="2023-01-01")["Close"]
    data2 = yf.download(stock2, start="2020-01-01", end="2023-01-01")["Close"]

    # Save the data to CSV files
    data1.to_csv(f"{stock1}_data.csv")
    data2.to_csv(f"{stock2}_data.csv")

    # Create a DataFrame to align both time series data
    combined_data = pd.DataFrame({"Stock1": data1, "Stock2": data2})

    # Compute the correlation
    correlation = combined_data.corr().iloc[0, 1]

    return correlation


# Example usage: Computing the correlation between Apple (AAPL) and Google (GOOGL)
apple_google_correlation = compute_correlation("AAPL", "GOOGL")
print("Correlation between Apple and Google stocks:", apple_google_correlation)
