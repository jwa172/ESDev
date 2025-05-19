from pathlib import Path
from dash import html, dcc
import dash_bootstrap_components as dbc

def make_dropdown(dropdown_id, options, value=None, label_text="", class_name="dropdown-dark mb-3"):
    """Creates a labeled dropdown with consistent styling."""
    return html.Div([
        html.Label(label_text, className="text-light mb-2"),
        dcc.Dropdown(
            id=dropdown_id,
            options=options,
            value=value,
            clearable=False,
            className=class_name
        )
    ])

def create_header_bar():
    """Top header with title and navigation."""
    return html.Div([
        html.Div("Historical Data Viewer", className="app-title"),
        dcc.Link("Back to Dashboard", href="/", className="btn btn-outline-light btn-sm"),
        html.Div(id="history-current-time", className="timestamp")
    ], className="header-bar")

def create_dashboard_container(current_folder):

    archive_files = []
    if current_folder:
        archive_path = Path(current_folder) / "data_archive"
        if archive_path.exists():
            archive_files = sorted(archive_path.glob("*.csv"), reverse=True)

    archive_dropdown = make_dropdown(
        dropdown_id='archive-file-selector',
        options=[{'label': f.name, 'value': str(f)} for f in archive_files],
        value=str(archive_files[0]) if archive_files else None,
        label_text="Select Archive File:",
    )

    # Make time window dropdown 'Time Window'
    time_window_dropdown = make_dropdown(
        dropdown_id='history-time-window',
        options=[
            {'label': 'All Data', 'value': 0},
            {'label': '1 min', 'value': 60},
            {'label': '5 min', 'value': 300},
            {'label': '10 min', 'value': 600},
            {'label': '30 min', 'value': 1800},
        ],
        value=0,
        label_text="Time Window:",
    )
    return html.Div([
            dbc.Row([
                dbc.Col(archive_dropdown, width=6),
                dbc.Col(time_window_dropdown, width=6),
            ]),
            dcc.Graph(id="historical-data-graph", style={'height': '70vh'}, config={'displayModeBar': False})
        ], className="dashboard-container")
    
# Defines and returns a Dash layout specifically for viewing historical sensor data
def historical_data_layout(folder_path):
    return html.Div([
        create_header_bar(),
        create_dashboard_container(folder_path)
    ])
