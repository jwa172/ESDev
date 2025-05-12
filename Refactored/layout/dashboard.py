from dash import html, dcc
import dash_bootstrap_components as dbc

def dashboard_layout():
    return html.Div([
        html.Div("Production Monitoring System", className="app-title"),
        html.Div([
            html.Button("Choose Folder", id="folder-button", className="btn btn-outline-light btn-sm me-2"),
            html.Button("Start Monitoring", id="toggle-button", n_clicks=0, className="btn btn-success btn-sm me-2"),
            html.Button("Stop Monitoring", id="stop-button", n_clicks=0, className="btn btn-danger btn-sm"),
            html.Button("Historical Data", id="history-button", className="btn btn-outline-light btn-sm me-2"),
        ], className="header-bar"),
        dcc.Graph(id="live-graph", style={'height': '40vh'})
    ])
