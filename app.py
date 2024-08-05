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


def get_color_by_index(index, total, colormap):
    norm = plt.Normalize(vmin=0, vmax=total-1)
    return cm.ScalarMappable(norm=norm, cmap=colormap).to_rgba(index)


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

    return tickerDate


def addROIStockPrice(stockPrice, df_meta):
    # add gain
    for _, row in df_meta.iterrows():
        if row['Ticker Symbol'] in stockPrice.keys():
            stockPrice[row['Ticker Symbol']
                       ]['ROI'] = row['ROI']


# Read the data from the CSV file
df = pd.read_csv('2020-2024.csv')
df_meta = pd.read_csv('company meta 2020-2024.csv')
# Convert 'Transaction Date' to datetime
df['Transaction Date'] = pd.to_datetime(
    df['Transaction Date'], format='%Y/%m/%d')

# Create traces for each action type
buy_trace = go.Scatter(
    x=df[df['Action'] == 'Market buy']['Transaction Date'],
    y=df[df['Action'] == 'Market buy']['Price / share'],
    mode='markers',
    name='Market buy',
    marker=dict(color='green', symbol='triangle-up', size=5),
    text=df[df['Action'] == 'Market buy']['Ticker'],
    hovertemplate='<b>Ticker:</b> %{text}<br><b>Date:</b> %{x}<br><b>Price / share:</b> %{y}<extra></extra>'
)

sell_trace = go.Scatter(
    x=df[df['Action'] == 'Market sell']['Transaction Date'],
    y=df[df['Action'] == 'Market sell']['Price / share'],
    mode='markers',
    name='Market sell',
    marker=dict(color='red', symbol='triangle-down', size=5),
    text=df[df['Action'] == 'Market sell']['Ticker'],
    hovertemplate='<b>Ticker:</b> %{text}<br><b>Date:</b> %{x}<br><b>Price / share:</b> %{y}<extra></extra>'
)

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
                value='white',
                labelStyle={'display': 'block', 'margin-bottom': '5px'}
            )
        ], style={'width': '24%'}),

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
        ], style={'width': '24%'}),

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
        ], style={'width': '24%'}),

        html.Div([
            html.Label('Y-axis Maximum Value:'),
            dcc.Input(
                id='default_y_max',
                type='number',
                value=1000,  # Default value for the y-axis maximum
                step=50,
                style={'width': '50px'}
            )
        ], style={'width': '24%'})
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
     Input('default_y_max', 'value')]
)
def update_chart(bg_color, moving_average, discrete_colormap, default_y_max):
    # Create the layout with the selected background color
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
    fig = go.Figure(data=[buy_trace, sell_trace], layout=layout)

    # Add traces for stock prices
    tickers = list(stocksPrice.keys())
    # num_tickers = len(tickers)
    ROI = [
        item['ROI']for item in stocksPrice.values()]
    minROI = min(ROI)
    maxROI = max(ROI)

    for __, ticker in enumerate(tickers):
        data = stocksPrice[ticker]
        color = []
        # if capital_map == 'ticker':
        #     color = get_color_by_index(index, num_tickers, colormap)
        # else:
        date = data['price'].index
        price = data['price']['50_day_MA' if '50_day_MA' in moving_average else 'Close']
        ROI = data['ROI']
        color = get_color_by_value(ROI,
                                   minROI, maxROI, colormap,
                                   -1 if discrete_colormap == 'continuous' else NUM_COLOR_BIN)
        fig.add_trace(go.Scatter(
            x=date,
            y=price,
            mode='lines',
            name=ticker,
            line=dict(
                color=f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})', width=1),
            hovertemplate=f"<b>Ticker:</b> {ticker}<br>"
            f"<b>Date:</b> %{{x|%Y-%m-%d}}<br>"
            f"<b>Price / share:</b> %{{y:.2f}}<br>"
            f"<b>ROI:</b> {ROI:.2f}<extra></extra>"
        ))

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
