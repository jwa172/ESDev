from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from layout.dashboard import dashboard_layout
from layout.historical import historical_data_layout
from callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=False, threaded=True, use_reloader=False, port=8050)
