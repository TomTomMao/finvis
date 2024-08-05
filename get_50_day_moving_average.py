import pandas as pd
import yfinance as yf


from datetime import datetime, timedelta


def get_50_day_moving_average(ticker, start_date, end_date):
    # Convert string dates to datetime objects
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate the date 100 days before the start_date to ensure enough trading days
    adjusted_start_date = start_date_dt - timedelta(days=100)

    # Fetch the stock data from the adjusted start date to the end date
    stock_data = yf.download(ticker, start=adjusted_start_date, end=end_date)

    # Check if the stock data is empty
    if stock_data.empty:
        return pd.DataFrame(), f"No data found for {ticker} between {adjusted_start_date.date()} and {end_date}"

    # Calculate the 50-day moving average
    stock_data['50_day_MA'] = stock_data['Close'].rolling(window=50).mean()

    # Filter the data to only include the range between start_date and end_date
    filtered_stock_data = stock_data.loc[start_date_dt:end_date_dt]

    return filtered_stock_data[['Close', '50_day_MA']], None