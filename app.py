from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from get_50_day_moving_average import get_50_day_moving_average

NUM_COLOR_BIN = 5
COLORBAR_X = 0.5
COLORBAR_Y = -0.25
COLORBAR_LEN = 0.75
DEFAULT_LINE_WIDTH = 1.3
DEFAULT_SHADING_TOP_OPACITY = 0.8
DEFAULT_SHADING_MIDPOINT = 0.44
DEFAULT_SHADING_MIDPOINT_OPACITY = 0.23
DEFAULT_MIN_MARKER_SIZE = 5
DEFAULT_MAX_MARKER_SIZE = 40
SHADING_OPACITY = 0.3
STOCK_NOT_SHOW = ['PBCT', 'LVGO', 'STOR', 'SBNY', 'WORK']
EXCHANGE_RATE_GBP_USD = {'2020-05-04': 1.2445,
                         '2020-08-25': 1.315, '2020-12-02': 1.3368, '2021-01-22': 1.3682}


class SizeMapper:
    def __init__(self, range, scale=(0, 10), log=False):
        """
        Initialize the SizeMapper with the desired range and scale.

        :param range: Tuple (min, max) representing the range of input values.
        :param scale: Tuple (min, max) representing the range of output sizes.
        :param log: Boolean indicating whether to apply logarithmic scaling.
        """
        self.min_val, self.max_val = range
        self.min_size, self.max_size = scale
        self.log = log

    def __call__(self, value):
        """
        Map the input value to the corresponding size.

        :param value: Input value to be scaled.
        :return: Scaled size.
        """
        # Ensure min_val and max_val are floats
        if isinstance(self.min_val, (int, float)) and isinstance(self.max_val, (int, float)):
            # Clip the value to be within the min and max range
            value = np.clip(value, self.min_val, self.max_val)

            # Apply logarithmic scaling if requested
            if self.log:
                value = np.log(value) if value > 0 else self.min_val

            # Normalize the value to [0, 1]
            normalized_value = (value - self.min_val) / \
                (self.max_val - self.min_val)

            # Scale the normalized value to the desired size range
            return self.min_size + normalized_value * (self.max_size - self.min_size)
        else:
            raise ValueError("min_val and max_val must be numeric types")


def get_dummy_trace(colormap, minROI, maxROI):
    original_colorscale = [
        [i / NUM_COLOR_BIN, f"rgb({int(colormap(i / NUM_COLOR_BIN)[0]*255)}, {int(colormap(
            i / NUM_COLOR_BIN)[1]*255)}, {int(colormap(i / NUM_COLOR_BIN)[2]*255)})"]
        for i in range(NUM_COLOR_BIN + 1)
    ]
    extended_colorscale = []
    for i in range(len(original_colorscale) - 1):
        start_val, start_color = original_colorscale[i]
        end_val, end_color = original_colorscale[i + 1]

        for step in range(NUM_COLOR_BIN):
            fraction = step / (NUM_COLOR_BIN - 1)
            value = start_val + fraction * (end_val - start_val)
            color = start_color
            extended_colorscale.append([round(value, 2), color])

    # Append the last color scale stop
    extended_colorscale.append(original_colorscale[-1])

    dummy_trace_continuous = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(
                title='ROI',
                titleside='bottom',
                orientation='h',
                x=COLORBAR_X,
                y=COLORBAR_Y,
                len=COLORBAR_LEN
            ),
            cmin=minROI,
            cmax=maxROI
        ),
        hoverinfo='none'
    )

    dummy_trace_discrete = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=extended_colorscale,
            showscale=True,
            colorbar=dict(
                title='ROI',
                titleside='bottom',
                orientation='h',
                x=COLORBAR_X,
                y=COLORBAR_Y,
                len=COLORBAR_LEN,
                tickvals=np.linspace(
                    minROI, maxROI, NUM_COLOR_BIN+1),
                ticktext=[f'{val:.2f}' for val in np.linspace(
                    minROI, maxROI, NUM_COLOR_BIN+1)]
            ),
            cmin=minROI,
            cmax=maxROI
        ),
        hoverinfo='none'
    )
    return dummy_trace_continuous, dummy_trace_discrete


def get_color_by_value(value, minValue, maxValue, colormap, num_bins=-1):
    '''
        num_bins: if =-1, use continuous color map, other wise use discrete
    '''
    norm = plt.Normalize(vmin=minValue, vmax=maxValue)
    normalized_value = norm(value)
    if num_bins == -1:
        return colormap(normalized_value)
    bins = np.linspace(0, 1, num_bins + 1)
    bin_index = np.digitize(normalized_value, bins) - 1
    bin_index = np.clip(bin_index, 0, num_bins - 1)

    return colormap(bins[bin_index])


def getStockPrice(df: pd.DataFrame):
    # sort df by date
    df_sorted = df.sort_values(by='Date', inplace=False)
    df_ticker_map = {}  # {'ticker': {minDate: , maxDate:, shares: }}
    for _, row in df_sorted.iterrows():
        if row['Action'] not in ['Market buy', 'Market sell']:
            continue
        shares = row['No. of shares'] if row['Action'] == 'Market buy' else - \
            row['No. of shares']
        if row['Ticker'] not in df_ticker_map:
            df_ticker_map[row['Ticker']] = {
                'minDate': row['Date'],
                'maxDate': row['Date'],
                'shares': shares
            }
        else:
            df_ticker_map[row['Ticker']]['maxDate'] = row['Date']
            df_ticker_map[row['Ticker']]['shares'] += shares
    # Update maxDate to today's date if shares equal zero

    sold_holding_tickers = set()
    current_holding_tickers = set()

    for ticker, data in df_ticker_map.items():
        if data['shares'] > 1:
            if ticker not in ['AAPL', 'AMAT', 'AMT', 'AMZN', 'AVGO', 'BKNG', 'CAT', 'COST', 'CRM', 'GOOGL', 'HD', 'HRZN', 'INTU', 'MA', 'MCO', 'NNN', 'O', 'SPG', 'SPGI', 'UNH', 'V', 'VICI']:
                print(ticker)
                print(df_ticker_map[ticker])
            data['maxDate'] = pd.Timestamp(datetime.now().date())
            current_holding_tickers.add((ticker))
        else:
            sold_holding_tickers.add(ticker)
    stockPrice = {}
    for ticker in df_ticker_map.keys():
        min_date = df_ticker_map[ticker]['minDate']
        max_date = df_ticker_map[ticker]['maxDate']
        if ticker in STOCK_NOT_SHOW:
            continue
        price_data, error = get_50_day_moving_average(
            ticker, min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
        if error:
            print(error)
        else:
            stockPrice[ticker] = {
                'min': min_date,
                'max': max_date,
                'price': price_data
            }

    return stockPrice, sold_holding_tickers, current_holding_tickers


def addROIStockPrice(stockPrice, df_meta):
    # add gain
    for _, row in df_meta.iterrows():
        if row['Ticker Symbol'] in stockPrice.keys():
            stockPrice[row['Ticker Symbol']
                       ]['ROI'] = row['ROI']
            print(row['ROI'])


def getAveragePriceOuter(df_meta: pd.DataFrame):
    ticker_average_price_map = {}
    for _, row in df_meta.iterrows():
        ticker_average_price_map[row['Ticker Symbol']
                                 ] = row['Average Price per Share USD']

    def getAveragePriceInner(ticker):
        assert type(ticker_average_price_map[ticker]) == float, f"ticker: {ticker}, {
            ticker_average_price_map[ticker]}, type: {type(ticker_average_price_map[ticker])}"
        return ticker_average_price_map[ticker]
    return getAveragePriceInner


def addStockPriceForDividend(stocksPrice, dividend_df):
    # Function to get stock price and 50-day moving average for a given row
    def get_price_and_moving_avg(row):
        ticker = row['Ticker']
        date = row['Date']

        # Check if the ticker is in the stocksPrice dictionary
        if ticker in stocksPrice:
            price_data = stocksPrice[ticker]['price']

            # Find the stock price on the dividend date
            if date in price_data.index:
                stock_price_on_date = price_data.loc[date, 'Close']
                moving_average_50 = price_data.loc[date, '50_day_MA']
            else:
                # If the exact date is not found, set as NaN
                stock_price_on_date = np.nan
                moving_average_50 = np.nan
        else:
            stock_price_on_date = np.nan
            moving_average_50 = np.nan

        return pd.Series([stock_price_on_date, moving_average_50])

    # Apply the function to each row and assign new columns to dividend_df
    dividend_df[['Stock Price on Date', '50-Day Moving Average']
                ] = dividend_df.apply(get_price_and_moving_avg, axis=1)

    return dividend_df


# df = pd.read_csv('2020-2024-split.csv')
df = pd.read_excel('Open Investment Data 2020-2024.xlsx',
                   '2020-2024 with Splits')
df['Date'] = pd.to_datetime(
    df['Date'], format='%Y/%m/%d')
# replace price with adjusted price if needed
df['Price / share'] = np.where(
    df['Price / share adjusted for split(s) (USD)'].isnull() | (
        df['Price / share adjusted for split(s) (USD)'] == ''),
    df['Price / share (USD)'],
    df['Price / share adjusted for split(s) (USD)']
)

df['Price / share'] = df.apply(
    lambda row: row['Price / share'] if row['Currency (Price / share)'] == 'USD' or row['Currency (Price / share)'] is np.nan
    else row['Price / share'] / 100 * EXCHANGE_RATE_GBP_USD[row['Date'].strftime('%Y-%m-%d')],
    axis=1
)

df['No. of shares'] = np.where(
    df['No. of shares adjusted for split(s)'].isnull() | (
        df['No. of shares adjusted for split(s)'] == ''),
    df['No. of shares'],
    df['No. of shares adjusted for split(s)']
)
# df_meta = pd.read_csv('company meta 2020-2024.csv')
df_meta = pd.read_excel(
    'Open Investment Data 2020-2024.xlsx', 'Company MetaData-2024')
# replace CWEN/A with CWEA
df_meta = df_meta.replace({'CWEN/A': 'CWEN'})

df_meta['Average Price per Share USD'] = np.where(df_meta['Average Price per Share after Split USD'].isnull(),
                                                  df_meta[
                                                      'Average Price per Share USD\n(=Total Purchase Amount / Total Number of Shares) '],
                                                  df_meta['Average Price per Share after Split USD'],
                                                  )
# replace na with 0
df_meta['Average Price per Share USD'] = df_meta['Average Price per Share USD'].apply(
    lambda x: 0 if pd.isna(x) else x
)
# calculate roi and replace na by 0
df_meta['ROI'] = (df_meta['Current Share Price USD\n (Yahoo & Google Finance)'] -
                  df_meta['Average Price per Share USD']) / df_meta['Average Price per Share USD']
df_meta['ROI'] = df_meta['ROI'].apply(
    lambda x: 0 if pd.isna(x) or np.isinf(x) else x)
df = df.replace({'CWEN/A': 'CWEN'})
df_meta.to_csv('df_meta_temp.csv')

getAveragePrice = getAveragePriceOuter(df_meta)
# Convert 'Date' to datetime
stocksPrice, sold_holding_tickers, current_holding_tickers = getStockPrice(df)
addROIStockPrice(stocksPrice, df_meta)

# Define a function to map ROI to colors
colormap = cm.get_cmap('RdYlGn')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # First row of controls
    html.Div([
        html.Div([
            html.Label('Background Color:'),
            dcc.RadioItems(
                id='bg-color',
                options=[
                    {'label': 'White Background', 'value': 'white'},
                    {'label': 'Black Background', 'value': 'black'}
                ],
                value='black',
                labelStyle={'display': 'inline-block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.66%'}),
        html.Div([
            html.Label('ROI Filter:'),
            dcc.RadioItems(
                id='roi-filter',
                options=[
                    {'label': 'all', 'value': 'all'},
                    {'label': 'positive', 'value': 'positive'},
                    {'label': 'negative', 'value': 'negative'}
                ],
                value='all',
                labelStyle={'display': 'inline-block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.66%'}),
        html.Div([
            html.Label('Holding Filter:'),
            dcc.RadioItems(
                id='holding_filter',
                options=[
                    {'label': 'all', 'value': 'all'},
                    {'label': 'current', 'value': 'current'},
                    {'label': 'sold', 'value': 'sold'}
                ],
                value='all',
                labelStyle={'display': 'inline-block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.66%'}),
        html.Div([
            html.Label('Moving Average:'),
            dcc.Checklist(
                id='moving_average',
                options=[{
                    'label': '50 day moving average', 'value': '50_day_MA'
                }],
                value=[],
                style={'width': '100%'}
            )
        ], style={'width': '16.66%'}),

        html.Div([
            html.Label('Color Map Type:'),
            dcc.RadioItems(
                id='discrete_colormap',
                options=[
                    {'label': 'Discrete color', 'value': 'discrete'},
                    {'label': 'Continuous color', 'value': 'continuous'}
                ],
                value='discrete',
                labelStyle={'display': 'inline-block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.66%'}),

        html.Div([
            html.Label('Action:'),
            html.Div([
                html.Div([
                    dcc.Checklist(
                        id='show_data_1',
                        options=[
                            {'label': 'Market Buy', 'value': 'market_buy'}],
                        value=['market_buy'],
                        style={'display': 'inline-block'}
                    )
                ]),

                # Second checklist
                html.Div([
                    dcc.Checklist(
                        id='show_data_2',
                        options=[{'label': 'Market Sell',
                                  'value': 'market_sell'}],
                        value=['market_sell'],
                        style={'display': 'inline-block'}
                    )
                ]),

                # Third checklist
                html.Div([
                    dcc.Checklist(
                        id='show_data_3',
                        options=[{'label': 'Dividend', 'value': 'dividend'}],
                        value=['dividend'],
                        style={'display': 'inline-block'}
                    )
                ])
            ], style={'display': 'flex', 'justify-content': 'start', 'flex-wrap': 'wrap'})
        ], style={'width': '16.66%'}),

    ], style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}),

    # second row for sliders
    html.Div([
        html.Div([
            html.Label('Y-axis Value:'),
            dcc.RangeSlider(
                0, 1500, 10,
                value=[0, 1000],  # Default value for the y-axis maximum
                id='y_range',
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '16.6%'}),
        html.Div([
            html.Label('Line Width:'),
            dcc.Slider(
                0.1, 2, 0.1, value=DEFAULT_LINE_WIDTH, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id='line_width',
            ),
            html.Div([
                dcc.Checklist(
                    id='shading',
                    options=[{
                        'label': 'shading', 'value': 'shading'
                    }],
                    value=['shading'],
                    style={'width': '100%'}
                ),
            ]),
        ], style={'width': '16.6%'}),
        html.Div([
            html.Label('shading top opacity:'),
                 dcc.Slider(0.01, 1, 0.01, value=DEFAULT_SHADING_TOP_OPACITY, marks=None, tooltip={
                            "placement": "bottom", "always_visible": True}, id='shading_top_opacity')
                 ], style={'width': '16.6%'}),
        html.Div([
            html.Label('shading midpoint:'),
            dcc.Slider(
                min=0.01,
                max=1,
                step=0.01,
                value=DEFAULT_SHADING_MIDPOINT,  # Default value for the y-axis maximum
                marks=None, tooltip={
                    "placement": "bottom", "always_visible": True},
                id='shading_midpoint',
            )
        ], style={'width': '16.6%'}),
        html.Div([
            html.Label('shading midpoint opacity:'),
            dcc.Slider(
                0.01,
                1,
                0.01,
                value=DEFAULT_SHADING_MIDPOINT_OPACITY,  # Default value for the y-axis maximum
                marks=None, tooltip={
                    "placement": "bottom", "always_visible": True},
                id='shading_midpoint_opacity',
            )
        ], style={'width': '16.6%'}),
        html.Div([
            html.Label('Glyph Size:'),
            dcc.RangeSlider(
                0, 100, 1,
                value=[DEFAULT_MIN_MARKER_SIZE, DEFAULT_MAX_MARKER_SIZE],  # Default value for the y-axis maximum
                id='glyph_size',
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '16.6%'}),
    ], style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}),
    # Third row for the chart
    html.Div([
        dcc.Graph(id='line-chart', style={'width': '100%'})
    ], style={'padding': '20px'}),
    html.Div([
        html.Label('Select Stock Ticker:'),
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[
                {'label': 'all', 'value': 'all'},
                # Add more tickers as needed
            ],
            value='all',  # Default selected ticker
            multi=False  # Set to True if you want to allow multiple selections
        )
    ], style={'width': '100%'}),
])

@app.callback(
    Output('ticker-dropdown', 'options'),
    [Input('roi-filter', 'value'),
     Input('holding_filter', 'value'),]
    # You can add more inputs here that affect the filtering logic
)
def update_dropdown(roi_filter, holding_filter):
    # Filter the DataFrame based ROI and holding options
    # Filter the data by ROI
    filtered_df_roi, roi_dict = filterByRoi(df, roi_filter)
    # print('roi_dict', roi_dict)
    # filter by holdings
    __, holding_set = filterByHolding(filtered_df_roi, holding_filter)
    # print('holding_set', holding_set)
    ticker_set = set(roi_dict.keys()) & holding_set
    ticker_options = [{'label': ticker, 'value': ticker} for ticker in sorted(list(ticker_set))]
    ticker_options_with_all = [{'label': 'all', 'value': 'all'}]
    ticker_options_with_all.extend(ticker_options)
    return ticker_options_with_all
@app.callback(
    Output('ticker-dropdown', 'value'),
    [Input('roi-filter', 'value'),
     Input('holding_filter', 'value')]
)
def reset_dropdown_value(roi_filter, holding_filter):
    # Whenever we update the options, reset the value to 'all'
    return 'all'
def filterByRoi(df, roi_filter):
    print('filterByRoiParams:', df, roi_filter)
    if roi_filter in ['positive', 'negative']:
        def condition(
            roi): return roi >= 0 if roi_filter == 'positive' else roi < 0
        roi_dict = {ticker: data['ROI'] for ticker,
                    data in stocksPrice.items() if condition(data['ROI'])}
    else:  # roi_filter == 'all'
        roi_dict = {ticker: data['ROI']
                    for ticker, data in stocksPrice.items()}
    filtered_df_roi = df[df['Ticker'].isin(roi_dict.keys())]  # filter by roi_dict
    return filtered_df_roi, roi_dict
def filterByHolding(df, holding_filter):
    if holding_filter == 'current':
        holding_set = current_holding_tickers
    elif holding_filter == 'sold':
        holding_set = sold_holding_tickers
    else:
        holding_set = current_holding_tickers.union(sold_holding_tickers)
    filtered_df_roi_holding = df[df['Ticker'].isin(holding_set)]
    return filtered_df_roi_holding, holding_set
#     return ticker_options
@app.callback(
    Output('line-chart', 'figure'),
    [Input('line-chart', 'restyleData'),
     Input('line-chart', 'relayoutData'),
     Input('bg-color', 'value'),
     Input('moving_average', 'value'),
     Input('discrete_colormap', 'value'),
     Input('y_range', 'value'),
     Input('line_width', 'value'),
     Input('roi-filter', 'value'),
     Input('holding_filter', 'value'),
     Input('shading', 'value'),
     Input('shading_top_opacity', 'value'),
     Input('shading_midpoint', 'value'),
     Input('shading_midpoint_opacity', 'value'),
     Input('show_data_1', 'value'),
     Input('show_data_2', 'value'),
     Input('show_data_3', 'value'),
     Input('ticker-dropdown', 'value'),
     Input('glyph_size', 'value')
     ]
)

def update_chart(restyle_data, relayout_data, bg_color, moving_average, discrete_colormap, y_range, line_width, roi_filter, holding_filter, shading, shading_top_opacity, shading_midpoint, shading_midpoint_opacity, show_data_1, show_data_2, show_data_3, selected_ticker,glyph_size):
    print('affected_traces', restyle_data)
    # Create the layout with the selected background color

    # add marker size based on the min and max value for the market buy and sell action
    # df['Total USD'] = df['No. of shares']*df['Price / share']
    # Define the list of all dividend-related actions
    dividend_actions = [
        'Dividend (Ordinary)',
        'Dividend (Return of capital non us)',
        'Dividend (Demerger)',
        'Dividend (Bonus)',
        'Dividend (Ordinary manufactured payment)',
        'Dividend (Dividends paid by us corporations)',
        'Dividend (Dividends paid by foreign corporations)',
        'Dividend (Dividend)'
    ]
    sell_buy_dividend_actions = dividend_actions + \
        ['Market sell', 'Market buy']
    df_sell_buy_dividend = df[df['Action'].isin(sell_buy_dividend_actions)]
    minTotal = df_sell_buy_dividend['Total USD'].min()
    maxTotal = df_sell_buy_dividend['Total USD'].max()
    print('minTotal=', minTotal, '($)')
    print('maxTotal=', maxTotal, '($)')
    size_mapper = SizeMapper(range=(minTotal, maxTotal), scale=(
        glyph_size[0], glyph_size[1]), log=False)
    df['Marker Size'] = df['Total USD'].apply(size_mapper)
    df.to_csv('df.csv')  # save for debug
    
    # Filter the data by ROI
    filtered_df_roi, roi_dict = filterByRoi(df, roi_filter)
    # print('roi_dict', roi_dict)
    # filter by holdings
    filtered_df_roi_holding, holding_set = filterByHolding(filtered_df_roi, holding_filter)
    # print('holding_set', holding_set)
    filtered_df = filtered_df_roi_holding
    ticker_set = set(roi_dict.keys()) & holding_set
    print('selected_ticker', selected_ticker)
    # Create traces for each action type
    buy_df = filtered_df[filtered_df['Action'] == 'Market buy']
    #apply ticker picker if picked
    if (selected_ticker != 'all'):
        buy_df = buy_df[buy_df['Ticker'] == selected_ticker]
    buy_trace = go.Scatter(
        x=buy_df['Date'],
        y=buy_df['Price / share'],
        mode='markers',
        name='Market buy',
        marker=dict(
            color='green',
            symbol='triangle-right',
            size=buy_df['Marker Size'],
            line=dict(width=0)
        ),
        text=buy_df['Ticker'],
        # Combine 'Total USD' and 'No. of shares' into customdata
        customdata=list(zip(buy_df['No. of shares'], buy_df['Total USD'])),
        hovertemplate='<b>Buy</b></br><b>Ticker:</b> %{text}<br>'
        '<b>Date:</b> %{x}<br>'
        '<b>Price / share:</b> %{y:.2f} ($)<br>'
        '<b>No. of shares:</b> %{customdata[0]:.2f}<br>'
        '<b>Total:</b> %{customdata[1]:.2f} ($)<extra></extra>'

    )
    sell_df = filtered_df[filtered_df['Action'] == 'Market sell']
    sell_df['color'] = sell_df.apply(
        lambda row: 'red' if row['Price / share'] < getAveragePrice(row['Ticker']) else 'green', axis=1)
    sell_df['Price Difference'] = sell_df.apply(
        lambda row: row['Price / share'] - getAveragePrice(row['Ticker']), axis=1)
    print('avgo AVG PURCHASE PRICE:', getAveragePrice('AVGO'))
    #apply ticker picker if picked
    if (selected_ticker != 'all'):
        sell_df = sell_df[sell_df['Ticker'] == selected_ticker]
    sell_trace = go.Scatter(
        x=sell_df['Date'],
        y=sell_df['Price / share'],
        mode='markers',
        name='Market sell',
        marker=dict(
            color=sell_df['color'],
            symbol='square',
            size=sell_df['Marker Size'],
            line=dict(width=0)
        ),
        text=sell_df['Ticker'],
        # Combine 'Total USD' and 'No. of shares' into customdata
        customdata=list(
            zip(sell_df['No. of shares'], sell_df['Total USD'], sell_df['Price Difference'])),
        hovertemplate='<b>Sell</b></br><b>Ticker:</b> %{text}<br>'
        '<b>Date:</b> %{x}<br>'
        '<b>Price / share:</b> %{y:.2f} ($)<br>'
        '<b>No. of shares:</b> %{customdata[0]:.2f}<br>'
        '<b>Price difference:</b> %{customdata[2]:.2f} ($)<br>'
        '<b>Total:</b> %{customdata[1]:.2f} ($)<extra></extra>'
    )

    dividend_df = filtered_df[filtered_df['Action'].isin(dividend_actions)]
    addStockPriceForDividend(stocksPrice, dividend_df)
    if (selected_ticker != 'all'):
        dividend_df = dividend_df[dividend_df['Ticker'] == selected_ticker]
    dividend_trace = go.Scatter(
        x=dividend_df['Date'],
        y=dividend_df['50-Day Moving Average' if '50_day_MA' in moving_average else 'Stock Price on Date'],
        mode='markers',
        name='Dividend',
        marker=dict(
            color='orange',
            symbol='cross',
            size=dividend_df['Marker Size'],
            line=dict(width=0)
        ),
        text=dividend_df['Ticker'],
        # Combine 'Total USD' and 'No. of shares' into customdata
        customdata=list(zip(
            dividend_df['No. of shares'], dividend_df['Total USD'], dividend_df['Price / share'], dividend_df['Action'])),
        hovertemplate='<b>%{customdata[3]}</b></br><b>Ticker:</b> %{text}<br>'
        '<b>Date:</b> %{x}<br>'
        '<b>Stock Price / share:</b> %{y:.2f} ($)<br>'
        '<b>Dividend Price / share:</b> %{customdata[2]:.2f} ($)<br>'
        '<b>No. of shares:</b> %{customdata[0]:.2f}<br>'
        '<b>Total:</b> %{customdata[1]:.2f} ($)<extra></extra>'
    )
    # color picked from: https://finance.yahoo.com/chart/COST#eyJsYXlvdXQiOnsiaW50ZXJ2YWwiOjYwLCJwZXJpb2RpY2l0eSI6NCwidGltZVVuaXQiOiJtaW51dGUiLCJjYW5kbGVXaWR0aCI6MS42Mjc5MzU3MjMxMTQ5NTY3LCJmbGlwcGVkIjpmYWxzZSwidm9sdW1lVW5kZXJsYXkiOnRydWUsImFkaiI6dHJ1ZSwiY3Jvc3NoYWlyIjp0cnVlLCJjaGFydFR5cGUiOiJtb3VudGFpbiIsImV4dGVuZGVkIjpmYWxzZSwibWFya2V0U2Vzc2lvbnMiOnt9LCJhZ2dyZWdhdGlvblR5cGUiOiJvaGxjIiwiY2hhcnRTY2FsZSI6ImxpbmVhciIsInN0dWRpZXMiOnsi4oCMdm9sIHVuZHLigIwiOnsidHlwZSI6InZvbCB1bmRyIiwiaW5wdXRzIjp7IlNlcmllcyI6InNlcmllcyIsImlkIjoi4oCMdm9sIHVuZHLigIwiLCJkaXNwbGF5Ijoi4oCMdm9sIHVuZHLigIwifSwib3V0cHV0cyI6eyJVcCBWb2x1bWUiOiIjMGRiZDZlZWUiLCJEb3duIFZvbHVtZSI6IiNmZjU1NDdlZSJ9LCJwYW5lbCI6ImNoYXJ0IiwicGFyYW1ldGVycyI6eyJjaGFydE5hbWUiOiJjaGFydCIsImVkaXRNb2RlIjp0cnVlfSwiZGlzYWJsZWQiOmZhbHNlfX0sInBhbmVscyI6eyJjaGFydCI6eyJwZXJjZW50IjoxLCJkaXNwbGF5IjoiQ09TVCIsImNoYXJ0TmFtZSI6ImNoYXJ0IiwiaW5kZXgiOjAsInlBeGlzIjp7Im5hbWUiOiJjaGFydCIsInBvc2l0aW9uIjpudWxsfSwieWF4aXNMSFMiOltdLCJ5YXhpc1JIUyI6WyJjaGFydCIsIuKAjHZvbCB1bmRy4oCMIl19fSwic2V0U3BhbiI6e30sIm91dGxpZXJzIjpmYWxzZSwiYW5pbWF0aW9uIjp0cnVlLCJoZWFkc1VwIjp7InN0YXRpYyI6dHJ1ZSwiZHluYW1pYyI6ZmFsc2UsImZsb2F0aW5nIjpmYWxzZX0sImxpbmVXaWR0aCI6MiwiZnVsbFNjcmVlbiI6dHJ1ZSwic3RyaXBlZEJhY2tncm91bmQiOnRydWUsImNvbG9yIjoiIzAwODFmMiIsInN5bWJvbHMiOlt7InN5bWJvbCI6IkNPU1QiLCJzeW1ib2xPYmplY3QiOnsic3ltYm9sIjoiQ09TVCIsInF1b3RlVHlwZSI6IkVRVUlUWSIsImV4Y2hhbmdlVGltZVpvbmUiOiJBbWVyaWNhL05ld19Zb3JrIiwicGVyaW9kMSI6MTYxMDExOTgwMCwicGVyaW9kMiI6MTcyNjY2NDQwMH0sInBlcmlvZGljaXR5Ijo0LCJpbnRlcnZhbCI6NjAsInRpbWVVbml0IjoibWludXRlIiwic2V0U3BhbiI6e319XSwicmFuZ2UiOnt9LCJyYW5nZVNsaWRlciI6ZmFsc2V9LCJldmVudHMiOnsiZGl2cyI6dHJ1ZSwic3BsaXRzIjp0cnVlLCJ0cmFkaW5nSG9yaXpvbiI6Im5vbmUiLCJzaWdEZXZFdmVudHMiOltdfSwicHJlZmVyZW5jZXMiOnsiY3VycmVudFByaWNlTGluZSI6dHJ1ZSwiZGlzcGxheUNyb3NzaGFpcnNXaXRoRHJhd2luZ1Rvb2wiOmZhbHNlLCJkcmFnZ2luZyI6eyJzZXJpZXMiOnRydWUsInN0dWR5IjpmYWxzZSwieWF4aXMiOnRydWV9LCJkcmF3aW5ncyI6bnVsbCwiaGlnaGxpZ2h0c1JhZGl1cyI6MTAsImhpZ2hsaWdodHNUYXBSYWRpdXMiOjMwLCJtYWduZXQiOmZhbHNlLCJob3Jpem9udGFsQ3Jvc3NoYWlyRmllbGQiOm51bGwsImxhYmVscyI6dHJ1ZSwibGFuZ3VhZ2UiOm51bGwsInRpbWVab25lIjoiQW1lcmljYS9OZXdfWW9yayIsIndoaXRlc3BhY2UiOjUwLCJ6b29tSW5TcGVlZCI6bnVsbCwiem9vbU91dFNwZWVkIjpudWxsLCJ6b29tQXRDdXJyZW50TW91c2VQb3NpdGlvbiI6ZmFsc2V9fQ--
    lineColor = 'rgb(26, 42, 54)' if bg_color == 'black' else 'rgb(226,229,232)'
    layout = go.Layout(
        title='Price/Share with Market Actions and Stock Prices',
        xaxis=dict(title='Date', gridcolor=lineColor, zerolinecolor=lineColor),
        yaxis=dict(title='Price (USD)', range=(y_range[0], y_range[1]),
                   gridcolor=lineColor, zerolinecolor=lineColor),
        height=700,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color='white' if bg_color == 'black' else 'black'),
        uirevision='constant'
    )

    # Create the figure and add traces
    fig = go.Figure(layout=layout)

    # Add traces for stock prices
    tickers = list(stocksPrice.keys())
    # num_tickers = len(tickers)
    ROI = [
        item['ROI']for item in stocksPrice.values()]
    minROI = min(ROI)
    maxROI = max(ROI)
    if (selected_ticker != 'all'):
        tickers = [selected_ticker]
    for __, ticker in enumerate(sorted(tickers)):
        data = stocksPrice[ticker]
        date = data['price'].index
        price = data['price']['50_day_MA' if '50_day_MA' in moving_average else 'Close']
        ROI = data['ROI']
        color = get_color_by_value(ROI,
                                   minROI, maxROI, colormap,
                                   -1 if discrete_colormap == 'continuous' else NUM_COLOR_BIN)

        if ticker in ticker_set:
            fig.add_trace(go.Scatter(
                x=date,
                y=price,
                mode='lines',
                name=ticker,
                line=dict(
                    color=f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})', width=line_width),
                fill='tozeroy' if 'shading' in shading else 'none',
                fillgradient=dict(
                    colorscale=[[0, f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, {shading_top_opacity})'],
                                [1 - shading_midpoint, f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, {shading_midpoint_opacity})'], [
                        1, f'rgba({color[0] * 255}, {color[1] * 255}, {color[2] * 255}, 0)']],
                    type='vertical',  # Gradient orientation
                    start=max(price),  # Gradient start position
                    stop=0  # Gradient stop position
                ),
                hovertemplate=f"<b>Ticker:</b> {ticker}<br>"
                f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                f"<b>Price / share:</b> %{{y:.2f}} ($)<br>"
                f"<b>ROI:</b> {ROI:.2f}<extra></extra>"
            ))

    show_data = show_data_1 + show_data_2 + show_data_3
    if 'market_buy' in show_data:
        fig.add_trace(buy_trace)
    if 'market_sell' in show_data:
        fig.add_trace(sell_trace)
    if 'dividend' in show_data:
        fig.add_trace(dividend_trace)

    dummy_trace_continuous, dummy_trace_discrete = get_dummy_trace(
        colormap, minROI, maxROI)
    # Hide legend for this trace
    dummy_trace_continuous.update(showlegend=False)
    dummy_trace_discrete.update(showlegend=False)  # Hide legend for this trace

    if discrete_colormap == 'continuous':
        fig.add_trace(dummy_trace_continuous)
    else:
        fig.add_trace(dummy_trace_discrete)

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
