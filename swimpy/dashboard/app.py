"""Plotly Dash related functionality.

The App class is the swimpy plugin that should be imported in the settings.
"""

# dash
from dash import Dash, html
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import dash_bootstrap_components as dbc

from modelmanager.utils import propertyplugin

from .layout import Layout
from .callbacks import Callbacks


@propertyplugin
class App:

    def __init__(self, project, **overwrite):
        self.project = project
        self.cache = diskcache.Cache("./cache")
        self.app = Dash(__name__,
            long_callback_manager=DiskcacheLongCallbackManager(self.cache),
            external_stylesheets=[dbc.themes.BOOTSTRAP],
        )
        self.layout = Layout(project)
        self.app.layout = self.layout.base()
        self.app.validation_layout = html.Div(
            [self.app.layout]+list(self.layout.tabs_content.values()))

        # add callbacks
        callbacks = Callbacks(self.project, self.layout)
        for f, kw in self.layout.callbacks.items():
            self.app.callback(**kw)(getattr(callbacks, f))
        for f, kw in self.layout.long_callbacks.items():
            self.app.long_callback(**kw)(getattr(callbacks, f))

        return

    def start(self, port=8054, debug=True):
        self.app.run_server(
            debug=debug,
            port=port,
        )

