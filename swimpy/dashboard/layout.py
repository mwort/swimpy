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

    parameter_groups = [
            ("Run parameters", "config_parameters", "nbyr  iyr".split()),
            ("Output parameters", "config_parameters", "isu1  isu2  isu3  isu4  isu5 is1   ih1   is2   ih2   is3   ih3   is4   ih4   is5   ih5   is6   ih6   is7   ih7".split()),
            ("Switches", "basin_parameters", "isc icn idlef intercep iemeth idvwk subcatch bResModule bWAM_Module bSnowModule radiation bDormancy".split()),
            ("Evaporation parameters", "basin_parameters", "ecal      thc       epco      ec1".split()),
            ("Groundwater parameters", "basin_parameters", "gwq0      abf0      bff".split()),
            ("Erosion parameters", "basin_parameters", "ekc0      prf       spcon     spexp".split()),
            ("Snow parameters", "basin_parameters", "tsnfall   tmelt     smrate   gmrate xgrad1    tgrad1    ulmax0    rnew snow1     storc1    stinco".split()),
            ("Routing parameters", "basin_parameters", "roc1      roc2      roc3      roc4".split()),
            ("Soil parameters", "basin_parameters", "sccor     prcor     rdcor".split()),
            ("Transmission loss parameters", "basin_parameters", "tlrch     evrch     tlgw maxup".split()),
        ]
    highlighted_parameters = [
        ("config_parameters", "iyr"),
        ("config_parameters", "nbyr"),
        ("basin_parameters", "sccor"),
        ("basin_parameters", "ecal"),
        ("basin_parameters", "roc2"),
        ("basin_parameters", "roc4"),
    ]
    
    tab_labels = ["Run model", "Parameters"] + list(output_tabs_functions.keys())

    callbacks = {
        "render_content": {
            "output": Output('tabs-content', 'children'),
            "inputs": Input('tabs-header', 'value'),
        },
        "store_parameters": {
            "state": {},  # parameter values, filled in parameter_tab
            "output": Output("btn-save-params", "children"),
            "inputs": {'n_clicks': Input("btn-save-params", "n_clicks")},
            "prevent_initial_call": True,
        }
    }

    long_callbacks = {
        "run_model": {
            "output": [
                Output("paragraph_id", "children"),
                Output("hydro_graph", "children"),
                ],
            "inputs": Input("btn-run", "n_clicks"),
            "state": [],
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

        # used as validation content, needed for initialisation of buttons and js libs
        self.tabs_init_content = {
            "tab-parameters": self.parameters_tab(),
            "tab-main": self.run_model_tab(),
            }
        self.tabs_init_content.update({ot: self.output_tab(ot) for ot in self.output_tabs_functions})
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
                            value='run-model-tab',
                            children=[
                                dcc.Tab(label=l, value=l.lower().replace(" ", "-")+'-tab')
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


        # build parameter inputs
        config_parameter_inputs = [
            dbc.Col(className="mt-3",
                    children=[
                        html.P(pname),
                        dbc.Input(
                            id=f"input-{group}-{pname}",
                            type="number",
                            value="%d" % (self.project.config_parameters[pname]),
                            className="w-75"
                        )])
            for group, pname in self.highlighted_parameters if group == "config_parameters"]
        basin_parameter_inputs = [
            dbc.Col(
                className="col-6 mt-2",
                children=[
                    html.P(pname),
                    dbc.Input(
                        step="0.1",
                        id=f"input-{group}-{pname}",
                        value="%d" % (self.project.basin_parameters[pname]),
                        type="number"
                    )])
            for group, pname in self.highlighted_parameters if group == "basin_parameters"]

        r = dbc.Row([
            dbc.Col([
                html.H3("Run parameters", className="pt-4"),
                dbc.Col(children=config_parameter_inputs),
                html.H3("Main model parameters", className="pt-4"),
                dbc.Row(className="w-75", children=basin_parameter_inputs),
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

        # parse parameters to run_model callback
        self.long_callbacks["run_model"]["state"] = [
            State(f"input-{group}-{pname}", "value")
            for group, pname in self.highlighted_parameters
        ]
        return r

    def parameter_input(self, name, value):
        ipt = dbc.Input(id="param-input-%s" % name, value=value, type="number")
        return ipt, dbc.InputGroup([dbc.InputGroupText(name), ipt], className="p-2")


    def parameters_tab(self):

        param_inputs, param_input_comps = {}, []
        for g, d, params in self.parameter_groups:
            param_dict = getattr(self.project, d)
            prmipt, prmcps = zip(*[self.parameter_input(n, param_dict[n]) for n in params])
            param_input_comps.append((g, prmcps))
            param_inputs.update(dict(zip(params, prmipt)))

        accordion = dbc.Accordion(
            [dbc.AccordionItem(params, title=t) for t, params in param_input_comps],
            start_collapsed=True,
            className="w-50")
        storebtn = dbc.Button("Save parameters", id="btn-save-params", className="m-4")
        confirmp = html.P(id="save-confirm")
        cont = dbc.Row([dbc.Col([storebtn, confirmp, accordion])])

        # parse inputs to store_parameters
        self.callbacks["store_parameters"]["state"] = {k: State(i.id, "value") for k, i in param_inputs.items()}
        return cont


    def output_tab(self, label):
        assert label in self.output_tabs_functions, "Undefined output tab label."
        cont = self.output_tabs_functions[label]
        tab = [dbc.Row([
                    dbc.Col(c(self.project) if hasattr(c, "__call__") else c)
                    for c in r])
                for r in cont]
        return tab
