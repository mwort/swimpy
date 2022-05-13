
import os.path as osp

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from dash.long_callback import DiskcacheLongCallbackManager

#import plotly.express as px
import diskcache

import pandas as pd
#import matplotlib.pyplot as plt
import swimpy

import dash_bootstrap_components as dbc


cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(__name__,
    long_callback_manager=long_callback_manager,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                html.P(id="paragraph_id", children=["Button not clicked"]),
                html.Progress(id="progress_bar"),
            ]
        ),
        dbc.Button(id="button_id", children="Run Job!"),
        dbc.Button(id="cancel_button_id", children="Cancel Running Job!"),
    ]
)

@app.long_callback(
    output=Output("paragraph_id", "children"),
    inputs=Input("button_id", "n_clicks"),
    running=[
        (Output("button_id", "disabled"), True, False),
        (Output("cancel_button_id", "disabled"), False, True),
        (
            Output("paragraph_id", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
        (
            Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
    ],
    cancel=[Input("cancel_button_id", "n_clicks")],
    progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    prevent_initial_call=True,
)
def callback(set_progress, n_clicks):
    import time
    import subprocess

    total = 10
    set_progress(("0", str(total)))
    try:
        process = subprocess.Popen(["sleep", "10"], close_fds=True)
    except:
        print("err", flush=True)
    stt = time.time()
    while process.poll() is None:
    #for _ in range(20):
        time.sleep(0.5)
        set_progress((str(int(time.time()-stt)), str(total)))
    set_progress(("10", str(total)))
    return [str(process.communicate())]


if __name__ == "__main__":
    app.run_server(debug=True,port=8054)