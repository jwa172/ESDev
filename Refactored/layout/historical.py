from dash import html, dcc

def historical_data_layout():
    return html.Div([
        html.Div("Historical Data Viewer", className="app-title"),
        dcc.Dropdown(id='archive-file-selector', className="dropdown-dark mb-3"),
        dcc.Dropdown(id="history-time-window", className="dropdown-dark mb-3"),
        dcc.Graph(id="historical-data-graph", style={'height': '70vh'})
    ])
