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
                      [graphs.table_daily_discharge_performance],
                      [graphs.plotly_station_daily_and_regime_discharge_component],
                    ],
        "Hydrotope": [[graphs.plotly_hydrotopes_daily_waterbalance]],
        "Maps": [[graphs.map_hydrotope_means("hydrotope_evapmean_gis")],
                 [graphs.map_hydrotope_means("hydrotope_gwrmean_gis")],
                 [graphs.map_hydrotope_means("hydrotope_pcpmean_gis")]],
        "Reservoirs": [[graphs.plotly_reservoir]],
        "Statistics": [[graphs.table_catchment_annual_waterbalance]],
    }

    parameter_groups = [
            ("Run parameters", "config_parameters", "nbyr  iyr".split()),
            ("Reservoir parameters", "reservoir_parameters", ['caphpp', 'rsveff', 'rsvevapc']),
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
        ("climate_parameters", "climate_input"),
        ("config_parameters", "iyr"),
        ("config_parameters", "nbyr"),
        ("basin_parameters", "sccor"),
        ("basin_parameters", "ecal"),
        ("basin_parameters", "roc2"),
        ("basin_parameters", "roc4"),
        ("run_parameters", "notes"),
    ]
    parameters_long_names = {
        "iyr": "Start year",
        "nbyr": "Number of years",
        "sccor": "Sat. conductivity",
        "ecal": "ETa coefficient",
        "roc2": "Routing quick",
        "roc4": "Routing slow",
    }
    tab_labels = ["Run model", "Parameters"] + list(output_tabs_functions.keys())

    callbacks = {
        "render_content": {
            "output": Output('tabs-content', 'children'),
            "inputs": [Input('tabs-header', 'value')],
        },
        "render_output": {
            "output": Output("output-body", "children"),
            "inputs": [Input("redraw-output", "n_clicks")],
            "state": [State('tabs-header', 'value'),
                      State("output-run-dropdown", "value"),
                      State("reference-run-dropdown", "value")],
            "prevent_initial_call": False,
            },
        "store_parameters": {
            "state": {},  # parameter values, filled in parameter_tab
            "output": Output("btn-save-params", "children"),
            "inputs": {'n_clicks': Input("btn-save-params", "n_clicks")},
            "prevent_initial_call": True,
        },
        "download_input": {
            "output": Output("download-input", "data"),
            "inputs": Input("btn-download-input", "n_clicks"),
            "prevent_initial_call": True,
        },
        "download_output": {
            "output": Output("download-output", "data"),
            "inputs": Input("btn-download-output", "n_clicks"),
            "prevent_initial_call": True,
        },
        "upload_project": {
            "output": Output('alert-div-upload', 'children'),
            "inputs": Input('upload-input', 'contents'),
            "state": [State('upload-input', 'filename'),
                      State('upload-input', 'last_modified')],
            "prevent_initial_call": True,
        },
        "reset_runs": {
            "output": Output('alert-div-reset', 'children'),
            "inputs": Input('reset-runs-button', "n_clicks"),
            "prevent_initial_call": True,
        },
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
        # save output and reference runs
        self.selected_output_run = None
        self.selected_reference_run = None

        # used as validation content, needed for initialisation of buttons and js libs
        self.tabs_init_content = {
            "tab-parameters": self.parameters_tab(),
            "tab-main": self.run_model_tab(),
            }
        self.tabs_init_content.update({ot: self.output_tab(ot) for ot in self.output_tabs_functions})

        return

    def base(self):
        itemkw = dict(outline=True, download=True, class_name="btn-outline-secondary")
        project_menu = html.Div([
            dbc.ButtonGroup([
                dbc.Button("Download input", id="btn-download-input", **itemkw),
                dbc.Button("Download output", id="btn-download-output", **itemkw),
                dbc.Button(dcc.Upload("Upload input", id="upload-input", accept=".zip", max_size=1024**3), **itemkw),
                dbc.Button("Reset runs", id="reset-runs-button", **itemkw),
            ], id="download-upload-menu", size="sm"),
            dcc.Download(id="download-input"),
            dcc.Download(id="download-output"),
        ],
            className="d-flex flex-row-reverse")
        c = dbc.Container(
            className="px-5",
            fluid=True,
            children=[
                html.Div(id='alert-div-upload'),
                html.Div(id='alert-div-reset'), 
                project_menu,
                html.Div(
                    className="d-flex flex-row",
                    children=[
                        html.Div(local_img(static_path("img/swim_logo_trans.png"), width="300px")),
                        dcc.Tabs(
                            id="tabs-header",
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


        # build parameter input
        climopts = {k: k + ", %i - %i" % se
                    for k, (_, se) in sorted(self.project.climate.options.items())}
        #climopts = sorted(self.project.climate.options.keys())
        config_parameter_inputs = [
                dbc.Row(className="pe-0", children=[
                    html.P("Climate input"),
                    dcc.Dropdown(climopts, list(climopts)[0], className="pe-0",
                                 id='input-climate_parameters-climate_input',
                                 optionHeight=60),
                ])
            ] + [
            dbc.Col(className="mt-2 col-6",
                    children=[
                        html.P(self.parameters_long_names[pname]),
                        dbc.Input(
                            id=f"input-{group}-{pname}",
                            type="number",
                            value="%d" % (self.project.config_parameters[pname]),
                        )])
            for group, pname in self.highlighted_parameters if group == "config_parameters"]
        basin_parameter_inputs = [
            dbc.Col(
                className="col-6 mt-2",
                children=[
                    html.P(self.parameters_long_names[pname]),
                    dbc.Input(
                        step="0.1",
                        id=f"input-{group}-{pname}",
                        value="%d" % (self.project.basin_parameters[pname]),
                        type="number"
                    )])
            for group, pname in self.highlighted_parameters if group == "basin_parameters"]

        r = dbc.Row([
            dbc.Col([
                html.H5("Main parameters", className="pt-4"),
                dbc.Row(children=config_parameter_inputs),
                dbc.Row(className="", children=basin_parameter_inputs),
                dbc.Row(className="", children=[dbc.Col([
                    html.P("Run notes:"),
                    dbc.Input(type="text", id="input-run_parameters-notes")],
                    className="mt-2")]),
                dbc.Col(
                    className="d-flex gap-3 mt-5",
                    children=[
                        dbc.Button("Run model", id="btn-run"),
                        dbc.Button("Cancel", id="btn-cancel-run"),
                    ]
                ),
                dbc.Progress(id="progress_bar", animated=True, striped=True, className="mt-3"),
                html.P(id="paragraph_id", children=["Not run"]),
            ], className="col-lg-3 col-md-12 order-md-2 order-lg-first"),

            dbc.Col([
                graphs.station_daily_discharge(self.project.station_daily_discharge, sim_start, sim_end),
                #graphs.hydrotopes_daily_waterbalance(self.project.hydrotope_daily_waterbalance, sim_start, sim_end),
                ],
                    id="hydro_graph", className="col-9"),
            dbc.Col(id="hydro_graph_progress", className="col-lg-9 col-md-12")
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

    def run_selectors(self, visible=True):
        runs = self.project.browser.runs.all().order_by("time").reverse()
        run_labels = [dict(value=r.id, label=f"{r}: {r.notes}") for r in runs]
        selectors = dbc.Row([
            dbc.Col(
            dbc.InputGroup([
                dbc.InputGroupText("Display run:", className="py-1"),
                dcc.Dropdown(run_labels, placeholder="Last run", id="output-run-dropdown",
                             value=self.selected_output_run),
            ], className="my-3 d-flex justify-content-end"),
            ),
            dbc.Col(dbc.Button("Update", id="redraw-output"),
                    class_name="col-1 my-3 d-flex justify-content-center"),
            dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Reference:", className="py-1"),
                dcc.Dropdown(run_labels, placeholder="None", id="reference-run-dropdown",
                             value=self.selected_reference_run),
            ], className="my-3 d-flex justify-content-start"),
        ]),
        ], class_name="" if visible else "invisible")
        return selectors

    def render_output(self, label, output_run=None, reference_run=None):
        print(label, output_run, reference_run)
        assert label in self.output_tabs_functions, "Undefined output tab label."
        cont = self.output_tabs_functions[label]
        self.selected_output_run = output_run
        self.selected_reference_run = reference_run
        if not output_run:
            lastr = self.project.browser.runs.last()
            kw = dict(run=lastr if lastr else self.project)
        else:
            kw = dict(run=self.project.browser.runs.get(id=output_run))
        if reference_run:
            kw["reference"] = self.project.browser.runs.get(id=reference_run)
        rows = [dbc.Row([
                    dbc.Col(c(self.project, **kw) if hasattr(c, "__call__") else c)
                    for c in r])
                for r in cont]
        return rows

    def output_tab(self, label):
        spinner = html.Div(html.Span(className="spinner-border spinner-border-sm", role="status"),
                           className="d-flex justify-content-center", style={"minHeight": "50vh"})
        body = html.Div(spinner, id="output-body")
        return [self.run_selectors(visible=label not in ("Hydrotope", "Reservoirs")), body]
