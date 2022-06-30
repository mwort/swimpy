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
        "Climate": [[graphs.plotly_basin_daily_weather]],
        "Discharge": [[graphs.plotly_station_daily_and_regime_discharge],
                      [graphs.plotly_station_daily_and_regime_discharge_component],
                    ],
        "Maps": [[graphs.map_hydrotope_means("hydrotope_evapmean_gis")],
                 [graphs.map_hydrotope_means("hydrotope_gwrmean_gis")],
                 [graphs.map_hydrotope_means("hydrotope_pcpmean_gis")]],
        "Statistics": [[graphs.table_catchment_annual_waterbalance],
                       [graphs.table_daily_discharge_performance],
                    ],
    }

    highlighted_parameters = [
        {"config_parameters": "iyr"},
        {"config_parameters": "nbyr"},
        {"basin_parameters": "sccor"},
        {"basin_parameters": "ecal"},
        {"basin_parameters": "bResModule"},
        {"basin_parameters": "bSnowModule"},
    ]
    
    tab_labels = ["Main", "Parameters"] + list(output_tabs_functions.keys())

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
                State("input-nbyr", "value"),
                State("input-iyr", "value"),
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
            "tab-main": self.run_model_tab(),
            }
        self.tabs_content.update(self.output_tabs())
        return

    def base(self):
        c = dbc.Container(
            className="p-5",
            fluid=True,
            children=[
                dbc.Col(
                    className="d-flex",
                    children=[
                        html.Div(
                            className="w-25",
                            children=[
                                html.Div(local_img(static_path("img/swim_logo_trans.png"), width="100%"))
                            ]
                        ),
                        dcc.Tabs(
                            id="tabs-header",
                            className="w-100",
                            value='tab-main',
                            children=[
                                dcc.Tab(label=l, value='tab-'+l.lower().replace(" ", "-"))
                                for l in self.tab_labels
                            ]
                        )
                    ]
                ),
                dbc.Row(id='tabs-content'),
        ])
        return c

    def run_model_tab(self):
        sim_start = self.project.config_parameters.start_date
        sim_end = self.project.config_parameters.end_date

        r = dbc.Row([
            dbc.Col([
                html.P("Config Parameters", style={"marginTop": 20, "fontSize": 24}),
                dbc.Col(
                    children=[dbc.Col(
                        className="mt-3",
                        children=[
                            html.P("%s" % (params['config_parameters']), style={"fontSize": 16}),
                            dbc.Input(
                                id="input-%s" % (params["config_parameters"]),
                                type="number",
                                value="%d" % (self.project.config_parameters[params["config_parameters"]]),
                                className="w-75"
                            )
                        ]
                    )
                    for params in self.highlighted_parameters if "config_parameters" in params.keys()  
                    ]
                ),
                html.P("Basin parameters", style={"fontSize": 24}, className="mt-5"),
                dbc.Row(
                    className="w-75",
                    children=[
                        dbc.Col(
                            className="col-6 mt-2",
                            children=[
                                html.P("%s :" % params["basin_parameters"], style={"fontSize": 14}),
                                dbc.Input(
                                    step="0.1",
                                    id="input-%s" % (params["basin_parameters"]),
                                    value="%d" % (self.project.basin_parameters[params['basin_parameters']]),
                                    type="number"
                                )
                            ]
                        )
                        for params in self.highlighted_parameters if "basin_parameters" in params.keys()
                    ]
                ),
                dbc.Col(
                    className="d-flex gap-3 mt-5",
                    children=[
                        dbc.Button("Run model", id="btn-run"),
                        dbc.Button("Cancel", id="btn-cancel-run"),
                    ]
                ),
                dbc.Progress(id="progress_bar", animated=True, striped=True, className="mt-3"),
                html.P(id="paragraph_id", children=["Not run"]),
            ], className="col-lg-4 col-md-12 order-md-2 order-lg-first"),
            dbc.Col(graphs.station_daily_discharge(self.project.station_daily_discharge, sim_start, sim_end),
                    id="hydro_graph", className="col-8"),
            dbc.Col(id="hydro_graph_progress", className="col-lg-8 col-md-12")
        ])
        return r

    def parameters_tab(self):
        items = [
            dbc.Col(
                className="col-3 mt-3",
                children=[
                    html.P("%s: " % (key), style={"fontSize": 14}),
                    dbc.Input(type="number", step="0.1", id="input-parameter-%s" % (key), value="%d" % (value))
                ]
            )
            for key, value in self.project.basin_parameters.items()
        ]
        c = dbc.Row(children=items)

        r = dbc.Row(style={"marginTop": 40}, children=[
            dbc.Col(
                className="col-4",
                children=[
                    html.P("Config parameters: ", style={"fontSize": 24}),
                    html.Div(
                        children=[html.Div(
                            className='mt-3',
                            children=[
                                html.P("%s" % (params['config_parameters']), style={"fontSize": 16}),
                                dbc.Input(
                                    id="input-%s" % (params['config_parameters']), 
                                    placeholder="years",
                                    type="number",
                                    value="%d" % (self.project.config_parameters[params['config_parameters']]),
                                    style={"width": "60%"}),
                            ]
                        )
                        for params in self.highlighted_parameters if 'config_parameters' in params.keys()
                        ]
                    ),
                ]
            ),
            dbc.Col(
                className="col-8", 
                children=[
                    html.P("Basin parameters:", style={"fontSize": 24}),
                    c
                ]
            ),
        ])
        return r 

    def output_tabs(self) -> dict:
        output_tab_labs = ['tab-'+l.lower().replace(" ", "-")
                           for l in self.output_tabs_functions.keys()]
        d = {ll: [dbc.Row([
                      dbc.Col(c(self.project) if hasattr(c, "__call__") else c)
                      for c in r]) for r in cont]
             for ll, (l, cont) in zip(output_tab_labs, self.output_tabs_functions.items())}
        return d
