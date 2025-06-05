import dash_bootstrap_components as dbc
from dash import dcc, html

"""
Main Dashboard Structure:
1. Header Bar
2. Toolbar
3. Dashboard Container
    - Punch Stats
    - Punch Status
    - Load Cell Forces
"""

# === 1. Header Bar ===
def create_header_bar():
    return html.Div([
            html.Div("Production Monitoring System", className="app-title"),
            html.Div([
                html.Button("Choose Folder", id="folder-button", className="btn btn-outline-light btn-sm me-2"),
                html.Button("Start Monitoring", id="toggle-button", n_clicks=0, className="btn btn-success btn-sm me-2"),
                html.Button("Stop Monitoring", id="stop-button", n_clicks=0, className="btn btn-danger btn-sm"),
                html.Button("Historical Data", id="history-button", className="btn btn-outline-light btn-sm me-2"),
            ]),
            html.Div(id="current-time", className="timestamp")
        ], className="header-bar")
    
# === 2. Toolbar ===
def create_toolbar(folder_path):

    FOLDER_DISPLAY = f"Data Source: {folder_path or 'Not selected'}"

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(id="selected-folder", className="text-light", children=[FOLDER_DISPLAY])
            ], width=4),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label("Time Window:", className="text-light me-2"),
                        dcc.Dropdown(
                            id="time-window-input",
                            options=[
                                {'label': '1 sec', 'value': 1000},
                                {'label': '10 sec', 'value': 10000},
                                {'label': '30 sec', 'value': 30000},
                                # {'label': '1 min', 'value': 60000},
                                # {'label': '10 min', 'value': 600000},
                                # {'label': '30 min', 'value': 1800000},
                            ],
                            value=1000,
                            clearable=False,
                            style={"width": "120px", "background-color": "#1f2937", "color": "white"},
                            className="dropdown-dark"
                        )
                    ], width="auto"),
                    dbc.Col([
                        dbc.Checklist(
                            id='show-tf-stats',
                            options=[{'label': ' Show Stats', 'value': 'tfstats'}],
                            value=['tfstats'],
                            switch=True
                        )
                    ], width="auto")
                ], justify="end")
            ], width=8)
        ])
    ], className="toolbar")

# === 3. Dashboard Sections ===

# Reusable card generator for punch stats
def create_punch_stats():
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Pre-Compression Punch Stats", className="card-header"),
                html.Div([
                    # Punch stats graph for pre-compression
                    dcc.Graph(
                        id="pre-compression-graph",
                        style={'height': '25vh'},
                        config={'displayModeBar': False}
                    ),
                    html.Div(id="pre-compression-stats", className="small-stats")
                ], className="card-body")
            ], className="card")
        ], width=4, className="grid-cell"),

        dbc.Col([
            html.Div([
                html.Div("Compression Punch Stats", className="card-header"),
                html.Div([
                    # Punch stats graph for compression
                    dcc.Graph(
                        id="compression-graph",
                        style={'height': '25vh'},
                        config={'displayModeBar': False}
                    ),
                    html.Div(id="compression-stats", className="small-stats")
                ], className="card-body")
            ], className="card")
        ], width=4, className="grid-cell"),

        dbc.Col([
            html.Div([
                html.Div("Ejection Punch Stats", className="card-header"),
                html.Div([
                    # Punch stats graph for ejection
                    dcc.Graph(
                        id="ejection-graph",
                        style={'height': '25vh'},
                        config={'displayModeBar': False}
                    ),
                    html.Div(id="ejection-stats", className="small-stats")
                ], className="card-body")
            ], className="card")
        ], width=4, className="grid-cell"),
    ])

# Helper function to create the eight punch status indicators
def create_punch_status():
    return dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("Punch Status", className="card-header"),
                        html.Div([
                            html.Div([
                                html.Span(id=f"punch-status-{i}", className="status-badge badge-normal", 
                                         children=[f"Punch {i}"]) for i in range(1, 9)
                            ], className="d-flex flex-wrap justify-content-between"),
                            html.Div(id="punch-status-indicator", className="small-stats mt-2")
                        ], className="card-body process-status-area")
                    ], className="card")
                ], width=12, className="grid-cell"),
            ])

# Helper function to create the live graph for load cell forces
def create_cell_forces():
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div("Load Cell Forces", className="card-header"),
                html.Div([
                    dcc.Graph(id='live-graph', style={'height': '40vh'}, config={'displayModeBar': False})
                ], className="card-body")
            ], className="card")
        ], width=12, className="grid-cell"),
    ])

# Function to create the main dashboard container
def create_dashboard_container():
    return html.Div([
        create_punch_stats(),
        create_punch_status(),
        create_cell_forces()
    ], className="dashboard-container")
    
# === 4. Main Layout Function ===
def dashboard_layout(folder_path=""):
    return html.Div([
        create_header_bar(),
        create_toolbar(folder_path),
        create_dashboard_container(),
    ])
