import os.path as osp

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.long_callback import DiskcacheLongCallbackManager

import plotly.express as px
import diskcache

import pandas as pd

import swimpy

import dash_bootstrap_components as dbc



def local_img(path, **kw):
    import base64
    encoded_image = base64.b64encode(open(path, 'rb').read()).decode()
    imgsrc = 'data:image/{0};base64,{1}'.format(osp.splitext(path)[1], encoded_image)
    return html.Img(src=imgsrc, **kw)


def station_daily_discharge_graph(station_daily_discharge, start, end):
    if station_daily_discharge is not None:
        q = (station_daily_discharge.stack()
             .reset_index()
             .set_axis(['time', "station", "discharge"], axis=1, inplace=False)
        )
        q.time = q.time.dt.to_timestamp()
        fig = px.line(q, x="time", y="discharge", color="station")
        graph = dcc.Graph(figure=fig)
    else:
        graph = dcc.Graph(figure=px.line([]))
    graph.figure.update_xaxes(
        range=[str(start), str(end)],
        autorange = False
        )
    return [graph]


swim = swimpy.Project("tests/project", stations=None)
sim_start = swim.config_parameters.start_date
sim_end = swim.config_parameters.end_date

output_tabs = {
    "Climate": [[1, 2], [3, 4]],
    "Maps": [[]],
    "Statistics": [[]],
}
output_tab_labs = ['tab-'+l.lower().replace(" ", "-") for l in output_tabs.keys()]
tabs = ["Run model", "Parameters"] + list(output_tabs.keys())

cache = diskcache.Cache("./cache")


app = Dash("SWIM dashboard",
    long_callback_manager=DiskcacheLongCallbackManager(cache),
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)


app.layout = dbc.Container([
    dbc.Row([
        html.Div(local_img("swim_logo_trans.png", width="200px")),
        dcc.Tabs(id="tabs-example-graph", value='tab-run-model',
            children=[dcc.Tab(label=l, value='tab-'+l.lower().replace(" ", "-")) for l in tabs])
        ]),
    dbc.Row(id='tabs-content-example-graph'),
])


tabs_content = {
    "tab-run-model":
        dbc.Row([
            dbc.Col([
                html.P("Save run parameters", style={"margin": 20}),
                dbc.InputGroup([
                    dbc.InputGroupText("Notes"),
                    dbc.Textarea(id="input-notes"),],
                    className="mb-3",
                ),
                dbc.InputGroup([
                    dbc.InputGroupText("Tags"),
                    dbc.Input(id="input-tags", placeholder="tag1 tag2 ..."),
                    ],
                    className="mb-3"),
                dbc.Button("Run model", id="btn-run", style={"margin": 20}),
                dbc.Button("Cancel", id="btn-cancel-run", style={"margin": 20}),
                dbc.Progress(id="progress_bar", animated=True, striped=True, style={"margin": 20}),
                html.P(id="paragraph_id", children=["Not run"]),
            ], className="col-3"),
            dbc.Col(station_daily_discharge_graph(swim.station_daily_discharge, sim_start, sim_end),
                    id="hydro_graph", className="col-9"),
            dbc.Col(id="hydro_graph_progress", className="col-9")
        ]),
    "tab-parameters":
        dbc.Accordion([
            dbc.AccordionItem([
                html.P("This is the content of the first section"),
                dbc.Button("Click here"),
            ],
            title="Climate parameters",
            ),
            dbc.AccordionItem([
                    html.P("This is the content of the second section"),
                    dbc.Button("Don't click me!", color="danger"),
                ],
                title="Evapotranspiration parameters",
            ),
            dbc.AccordionItem(
                "This is the content of the third section",
                title="Groundwater parameters",
            ),
        ])
}
tabs_content.update(
    {ll: [dbc.Row([dbc.Col(c) for c in r]) for r in cont]
    for ll, (l, cont) in zip(output_tab_labs, output_tabs.items())}
)
app.validation_layout = html.Div([app.layout]+list(tabs_content.values()))

@app.callback(Output('tabs-content-example-graph', 'children'),
              Input('tabs-example-graph', 'value'))
def render_content(tab):
    return html.Div(tabs_content[tab])


@app.long_callback(
    output=[
        Output("paragraph_id", "children"),
        Output("hydro_graph", "children"),
        ],
    inputs=Input("btn-run", "n_clicks"),
    state=[
        State("input-tags", "value"),
        State("input-notes", "value"),
    ],
    running=[
        (Output("btn-run", "disabled"), True, False),
        (Output("btn-cancel-run", "disabled"), False, True),
        (Output("paragraph_id", "style"),
            {"visibility": "hidden"},
            {"visibility": "visible"},
        ),
        (Output("hydro_graph", "style"), {"display": "none"}, {"display": "block"}),
    ],
    cancel=[Input("btn-cancel-run", "n_clicks")],
    progress=[
        Output("progress_bar", "value"),
        Output("progress_bar", "max"),
        Output("progress_bar", "label"),
        Output("hydro_graph_progress", "children"),
        ],
    progress_default=["0", "10", "", ""],
    prevent_initial_call=True,
)
def run_model(set_progress, n_clicks, tags, notes):
    import time
    from subprocess import PIPE, Popen

    ndays = (sim_end - sim_start).days

    swimcommand = [swim.swim, swim.projectdir+'/']
    process = Popen(swimcommand, stdout=PIPE, stderr=PIPE)

    prog = 0
    q = None
    while process.poll() is None:
        time.sleep(1)
        # reading discharge might fail due to incomplete lines
        try:
            q = swim.station_daily_discharge
            prog = len(q)
        except Exception:
            pass
        qgraph = station_daily_discharge_graph(q, sim_start, sim_end)
        set_progress((str(prog), str(ndays), "%1.0f%%" % (prog*100/ndays), qgraph))
    # make sure progress bar and graph are complete
    set_progress((str(ndays-1), str(ndays), "Saving...", qgraph))
    run = swim.save_run(tags=tags or "", notes=notes or "")
    qgraph = station_daily_discharge_graph(swim.station_daily_discharge, sim_start, sim_end)
    stdoutdata, stderrdata = process.communicate()
    return [stdoutdata.decode(), qgraph]


if __name__ == '__main__':
    app.run_server(
        debug=True,
        port=8054,
)