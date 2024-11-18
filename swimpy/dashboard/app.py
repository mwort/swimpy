"""Plotly Dash related functionality.

The App class is the swimpy plugin that should be imported in the settings.
"""
import os
# dash
from dash import Dash, html
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import dash_bootstrap_components as dbc

from modelmanager.utils import propertyplugin

from .layout import Layout
from .callbacks import Callbacks


DASHBOARD_BASE_URL = os.environ.get("DASHBOARD_BASE_URL", "/swim-dashboard/")
DASHBOARD_USERS = os.environ.get("DASHBOARD_USERS", None)


@propertyplugin
class App:

    def __init__(self, project, **overwrite):
        self.project = project
        self.cache = diskcache.Cache("./cache")
        self.app = Dash(__name__,
            title="SWIM",
            url_base_pathname=DASHBOARD_BASE_URL,
            serve_locally=True,
            prevent_initial_callbacks=False,
            long_callback_manager=DiskcacheLongCallbackManager(self.cache),
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=False,
        )
        self.layout = Layout(project)
        self.app.layout = self.layout.base()
        # needed for initialisation of buttons and js libs
        self.app.validation_layout = html.Div(
             [self.app.layout]+list(self.layout.tabs_init_content.values()))

        # add callbacks
        callbacks = Callbacks(self.project, self.layout)
        for f, kw in self.layout.callbacks.items():
            self.app.callback(**kw)(getattr(callbacks, f))
        for f, kw in self.layout.long_callbacks.items():
            self.app.long_callback(**kw)(getattr(callbacks, f))

        return

    def start(self, port=8054, users=None, debug=False, host='0.0.0.0'):

        if users is None and DASHBOARD_USERS:
            users = dict([tuple(u.split(":")) for u in DASHBOARD_USERS.split()])

        if users:
            import dash_auth
            self.auth = dash_auth.BasicAuth(self.app, users)

        self.app.run_server(
            debug=debug,
            port=port,
            host=host,
        )

