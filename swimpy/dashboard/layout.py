"""
Plotly Dash layout declarations.
"""

from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from . import graphs
from .utils import local_img, static_path


class Layout:

    output_tabs_functions = {
        "Climate": [[graphs.basin_daily_weather]],
        "Maps": [[]],
        "Statistics": [[]],
    }

    parameter_groups = {
        "Evapotranspiration": ["basin:ecal", "basin:thc"],
        "Routing": ["basin:roc2", "basin:roc4"],
        "Snow": ["basin:tsnfall", "basin:tmeld", "basin:smrate"],
    }
    
    tab_labels = ["Run", "Parameters"] + list(output_tabs_functions.keys())

    callbacks = {
        "render_content": {
            "output": Output('tabs-content', 'children'),
            "inputs": Input('tabs-header', 'value'),
        },
    }

    long_callbacks = {
        "run_model": {
            "output": [
                Output("paragraph_id", "children"),
                Output("hydro_graph", "children"),
                ],
            "inputs": Input("btn-run", "n_clicks"),
            "state": [
                State("input-tags", "value"),
                State("input-notes", "value"),
            ],
            "running": [
                (Output("btn-run", "disabled"), True, False),
                (Output("btn-cancel-run", "disabled"), False, True),
                (Output("paragraph_id", "style"),
                    {"visibility": "hidden"},
                    {"visibility": "visible"},
                ),
                (Output("hydro_graph", "style"), {"display": "none"}, {"display": "block"}),
            ],
            "cancel": [Input("btn-cancel-run", "n_clicks")],
            "progress": [
                Output("progress_bar", "value"),
                Output("progress_bar", "max"),
                Output("progress_bar", "label"),
                Output("hydro_graph_progress", "children"),
                ],
            "progress_default": ["0", "10", "", ""],
            "prevent_initial_call": True,
        }
    }

    def __init__(self, project, **overwrite):
        self.project = project
        self.tabs_content = {
            "tab-parameters": self.parameters_tab(),
            "tab-run": self.run_model_tab(),
            }
        self.tabs_content.update(self.output_tabs())
        return

    def base(self):
        c = dbc.Container([
            dbc.Row([
                html.Div(local_img(static_path("img/swim_logo_trans.png"), width="200px")),
                dcc.Tabs(id="tabs-header", value='tab-run',
                    children=[dcc.Tab(label=l, value='tab-'+l.lower().replace(" ", "-"))
                              for l in self.tab_labels])
                ]),
            dbc.Row(id='tabs-content'),
        ])
        return c

    def run_model_tab(self):
        sim_start = self.project.config_parameters.start_date
        sim_end = self.project.config_parameters.end_date

        r = dbc.Row([
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
            dbc.Col(graphs.station_daily_discharge(self.project.station_daily_discharge, sim_start, sim_end),
                    id="hydro_graph", className="col-9"),
            dbc.Col(id="hydro_graph_progress", className="col-9")
        ])
        return r

    def parameters_tab(self):
        items = [
            dbc.AccordionItem([
                dbc.InputGroup([
                    dbc.InputGroupText(i.split(":")[1]),
                    dbc.Input(id="input-parameter-%s" % (i.title()))],
                    className="mb-3")
                for i in pars
            ], title=t)
            for t, pars in self.parameter_groups.items()
        ]
        c = dbc.Accordion(items)
        return c

    def output_tabs(self) -> dict:
        output_tab_labs = ['tab-'+l.lower().replace(" ", "-")
                           for l in self.output_tabs_functions.keys()]
        d = {ll: [dbc.Row([
                      dbc.Col(c(self.project) if hasattr(c, "__call__") else c)
                      for c in r]) for r in cont]
             for ll, (l, cont) in zip(output_tab_labs, self.output_tabs_functions.items())}
        return d
