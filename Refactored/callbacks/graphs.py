from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from utils.archiver import DataArchiver

archiver = None

def register_graph_callbacks(app):
    global archiver

    @app.callback(
        Output("live-graph", "figure"),
        [Input("toggle-button", "n_clicks")],
        [State("folder-path", "data")]
    )
    def update_graph(n_clicks, folder_path):
        global archiver
        if folder_path and (archiver is None or archiver.db_folder.parent != folder_path):
            archiver = DataArchiver(folder_path, save_interval_seconds=300)

        df = pd.DataFrame({
            'timestamp': pd.date_range(start='now', periods=100, freq='s'),
            'value': range(100)
        })

        if archiver:
            archiver.save_data(df)

        return {
            'data': [go.Scatter(x=df['timestamp'], y=df['value'])],
            'layout': go.Layout(title="Live Data", plot_bgcolor='#1f2937', paper_bgcolor='#1f2937', font={'color': '#d1d5db'})
        }