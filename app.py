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
MIN_MARKER_SIZE = 5
MAX_MARKER_SIZE = 40
SHADING_OPACITY = 0.3


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


def getStockPrice(df):
    df_ticker_date = df.groupby('Ticker').agg({
        'Transaction Date': ['min', 'max']
    })
    minDateDict = df_ticker_date['Transaction Date'].to_dict()['min']
    maxDateDict = df_ticker_date['Transaction Date'].to_dict()['max']
    tickerDate = {}
    for ticker in df_ticker_date.index:
        min_date = minDateDict[ticker]
        max_date = maxDateDict[ticker]
        price_data, error = get_50_day_moving_average(
            ticker, min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
        if error:
            print(error)
        else:
            tickerDate[ticker] = {
                'min': min_date,
                'max': max_date,
                'price': price_data
            }
    print(tickerDate['INTU']['max'])
    print(tickerDate['INTU']['price'])
    return tickerDate


def addROIStockPrice(stockPrice, df_meta):
    # add gain
    for _, row in df_meta.iterrows():
        if row['Ticker Symbol'] in stockPrice.keys():
            stockPrice[row['Ticker Symbol']
                       ]['ROI'] = row['ROI']


df = pd.read_csv('2020-2024.csv')
df_meta = pd.read_csv('company meta 2020-2024.csv')
# Convert 'Transaction Date' to datetime
df['Transaction Date'] = pd.to_datetime(
    df['Transaction Date'], format='%Y/%m/%d')
stocksPrice = getStockPrice(df)
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
                labelStyle={'display': 'block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.6%'}),
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
                labelStyle={'display': 'block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.6%'}),
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
        ], style={'width': '16.6%'}),

        html.Div([
            html.Label('Color Map Type:'),
            dcc.RadioItems(
                id='discrete_colormap',
                options=[
                    {'label': 'Discrete color', 'value': 'discrete'},
                    {'label': 'Continuous color', 'value': 'continuous'}
                ],
                value='discrete',
                labelStyle={'display': 'block', 'margin-bottom': '5px'}
            )
        ], style={'width': '16.6%'}),

        html.Div([
            html.Label('Y-axis Maximum Value:'),
            dcc.Input(
                id='default_y_max',
                type='number',
                value=1000,  # Default value for the y-axis maximum
                step=50,
                style={'width': '50px'}
            )
        ], style={'width': '16.6%'}),
        html.Div([
            html.Label('Line Width:'),
            dcc.Input(
                id='line_width',
                type='number',
                value=DEFAULT_LINE_WIDTH,  # Default value for the y-axis maximum
                step=0.1,
                style={'width': '50px'}
            ),
            html.Div([
                dcc.Checklist(
                    id='shading',
                    options=[{
                        'label': 'shading', 'value': 'shading'
                    }],
                    value=[],
                    style={'width': '100%'}
                )
            ], style={'margin-top': '10px'})

        ], style={'width': '16.6%'})
    ], style={'display': 'flex', 'justify-content': 'space-between', 'flex-wrap': 'wrap'}),

    # Second row for the chart
    html.Div([
        dcc.Graph(id='line-chart', style={'width': '100%'})
    ], style={'padding': '20px'})
])


@app.callback(
    Output('line-chart', 'figure'),
    [Input('bg-color', 'value'),
     Input('moving_average', 'value'),
     Input('discrete_colormap', 'value'),
     Input('default_y_max', 'value'),
     Input('line_width', 'value'),
     Input('roi-filter', 'value'),
     Input('shading', 'value')
     ]
)
def update_chart(bg_color, moving_average, discrete_colormap, default_y_max, line_width, roi_filter, shading):
    # Create the layout with the selected background color

    # add marker size based on the min and max value for the market buy and sell action
    minTotal = df[(df['Action'] ==
                  'Market buy') | (df['Action'] == 'Market sell')]['Total'].min()
    maxTotal = df[(df['Action'] ==
                  'Market buy') | (df['Action'] == 'Market sell')]['Total'].max()
    print('minTotal=', minTotal)
    print('maxTotal=', maxTotal)
    size_mapper = SizeMapper(range=(minTotal, maxTotal), scale=(
        MIN_MARKER_SIZE, MAX_MARKER_SIZE), log=False)
    df['Marker Size'] = df['Total'].apply(size_mapper)
    df.to_csv('df.csv')  # save for debug
    # Filter the data by ROI
    if roi_filter in ['positive', 'negative']:
        def condition(
            roi): return roi >= 0 if roi_filter == 'positive' else roi < 0
        roi_dict = {ticker: data['ROI'] for ticker,
                    data in stocksPrice.items() if condition(data['ROI'])}
    else:  # roi_filter == 'all'
        roi_dict = {ticker: data['ROI']
                    for ticker, data in stocksPrice.items()}
    filtered_df = df[df['Ticker'].isin(roi_dict.keys())]

    # Create traces for each action type
    buy_trace = go.Scatter(
        x=filtered_df[filtered_df['Action'] ==
                      'Market buy']['Transaction Date'],
        y=filtered_df[filtered_df['Action'] == 'Market buy']['Price / share'],
        mode='markers',
        name='Market buy',
        marker=dict(
            color='green',
            symbol='triangle-up',
            size=filtered_df[filtered_df['Action']
                             == 'Market buy']['Marker Size'],
            line=dict(width=0)
        ),
        text=filtered_df[filtered_df['Action'] == 'Market buy']['Ticker'],
        customdata=filtered_df[filtered_df['Action'] == 'Market buy']['Total'],
        hovertemplate='<b>Buy</b></br><b>Ticker:</b> %{text}<br>'
        '<b>Date:</b> %{x}<br>'
        '<b>Price / share:</b> %{y}<br>'
        '<b>Total:</b> %{customdata}<extra></extra>'

    )

    sell_trace = go.Scatter(
        x=filtered_df[filtered_df['Action'] ==
                      'Sell']['Transaction Date'],
        y=filtered_df[filtered_df['Action'] == 'Market sell']['Price / share'],
        mode='markers',
        name='Market sell',
        marker=dict(
            color='red',
            symbol='triangle-down',
            size=filtered_df[filtered_df['Action']
                             == 'Market buy']['Marker Size'],
            line=dict(width=0)
        ),
        text=filtered_df[filtered_df['Action'] == 'Market sell']['Ticker'],
        customdata=filtered_df[filtered_df['Action']
                               == 'Market sell']['Total'],
        hovertemplate='<b>Market Sell</b></br><b>Ticker:</b> %{text}<br>'
        '<b>Date:</b> %{x}<br>'
        '<b>Price / share:</b> %{y}<br>'
        '<b>Total:</b> %{customdata}<extra></extra>'
    )
    layout = go.Layout(
        title='Price/Share with Market Actions and Stock Prices',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price', range=(0, default_y_max)),
        height=800,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color='white' if bg_color == 'black' else 'black')
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

    for __, ticker in enumerate(tickers):
        data = stocksPrice[ticker]
        date = data['price'].index
        price = data['price']['50_day_MA' if '50_day_MA' in moving_average else 'Close']
        ROI = data['ROI']
        color = get_color_by_value(ROI,
                                   minROI, maxROI, colormap,
                                   -1 if discrete_colormap == 'continuous' else NUM_COLOR_BIN)
        if ticker in roi_dict:
            fig.add_trace(go.Scatter(
                x=date,
                y=price,
                mode='lines',
                name=ticker,
                line=dict(
                    color=f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})', width=line_width),
                fill='tozeroy' if 'shading' in shading else 'none',
                fillcolor=f'rgba({color[0]*255}, {color[1]
                                                  * 255}, {color[2]*255}, {SHADING_OPACITY})',
                hovertemplate=f"<b>Ticker:</b> {ticker}<br>"
                f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
                f"<b>Price / share:</b> %{{y:.2f}}<br>"
                f"<b>ROI:</b> {ROI:.2f}<extra></extra>"
            ))

    fig.add_trace(buy_trace)
    fig.add_trace(sell_trace)

    dummy_trace_continuous, dummy_trace_discrete = get_dummy_trace(
        colormap, minROI, maxROI)

    if discrete_colormap == 'continuous':
        fig.add_trace(dummy_trace_continuous)
    else:
        fig.add_trace(dummy_trace_discrete)
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
