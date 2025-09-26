# src/dashboard/app.py
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import pandas as pd
from database.db_manager import CricketDatabaseManager

app = dash.Dash(__name__)
db_manager = CricketDatabaseManager()

app.layout = html.Div([
    html.H1("Cricket Trading Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Performance Overview", value="overview"),
        dcc.Tab(label="Recent Trades", value="trades"),
        dcc.Tab(label="Model Performance", value="models"),
    ]),

    html.Div(id="tab-content")
])


@app.callback(Output('tab-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'overview':
        return html.Div([
            dcc.Graph(id='pnl-chart'),
            dcc.Graph(id='win-rate-chart')
        ])
    elif tab == 'trades':
        return html.Div([
            html.H3("Recent Trading Activity"),
            html.Div(id='trades-table')
        ])
    elif tab == 'models':
        return html.Div([
            html.H3("ML Model Performance"),
            html.Div(id='model-metrics')
        ])


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)