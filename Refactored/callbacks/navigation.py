from dash.dependencies import Input, Output
from layout.dashboard import dashboard_layout
from layout.historical import historical_data_layout

def register_navigation_callbacks(app):
    @app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/history':
            return historical_data_layout()
        return dashboard_layout()