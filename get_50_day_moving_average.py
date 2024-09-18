import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def get_50_day_moving_average(ticker, start_date, end_date):
    # Convert string dates to datetime objects
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Define the directory and ensure it exists
    directory = "stocksprice"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate a filename based on the ticker and date range
    file_name = os.path.join(directory, f"{ticker}_{start_date}_{
                             end_date}_50_day_MA.csv")

    # Check if the file already exists
    if os.path.exists(file_name):
        print(f"Loading data from {file_name}")
        return pd.read_csv(file_name, index_col=0, parse_dates=True), None

    # Calculate the date 100 days before the start_date to ensure enough trading days
    adjusted_start_date = start_date_dt - timedelta(days=100)

    # Fetch the stock data from the adjusted start date to the end date
    stock_data = yf.download(
        ticker, start=adjusted_start_date, end=end_date_dt + timedelta(days=1))

    # Check if the stock data is empty
    if stock_data.empty:
        return pd.DataFrame(), f"No data found for {ticker} between {adjusted_start_date.date()} and {end_date}"

    # Calculate the 50-day moving average
    stock_data['50_day_MA'] = stock_data['Close'].rolling(window=50).mean()
    # Filter the data to only include the range between start_date and end_date
    filtered_stock_data = stock_data.loc[start_date_dt:end_date_dt]
    
    # replace empty value by the mean value of the previous values if there is less than 50 values
    i = 0
    sum_price = 0
    replaced = []
    for i in range(0, min(49, filtered_stock_data.shape[0])):
        divider = i + 1
        sum_price += filtered_stock_data.iloc[i]['Close']
        if pd.isna(filtered_stock_data.iloc[i]['50_day_MA']):
            filtered_stock_data.iloc[i, filtered_stock_data.columns.get_loc('50_day_MA')] = sum_price / divider
            replaced.append(i)
    
    # Save the filtered data to a CSV file in the stocksprice folder
    filtered_stock_data[['Close', '50_day_MA']].to_csv(file_name)
    print(f"Data saved to {file_name}")
    if len(replaced) > 0:
        print(f"stock {ticker} does not have enough data for 50 day moving average, these days' data are calculated differently" ,replaced)
    return filtered_stock_data[['Close', '50_day_MA']], None
