from .navigation import register_navigation_callbacks
from .graphs import register_graph_callbacks

def register_callbacks(app):
    register_navigation_callbacks(app)
    register_graph_callbacks(app)