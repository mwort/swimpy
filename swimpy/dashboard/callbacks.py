"""
Plotly Dash callback functions with access to the swimpy project and the Dash layout.

- The input/output declarations to these functions/methods are provided as class attributes
  in the layout.Layout class, since they contain layout knowledge. 
- The number of arguments (expt. self) needs to match the output + state declarations in
  the above above class attributes.
"""

from dash import html
import dash_bootstrap_components as dbc
import datetime as dt

from . import graphs


class Callbacks:

    def __init__(self, project, layout, **overwrite):
        self.project = project
        self.layout = layout

    def render_content(self, tab):

        isoutput = tab.replace("-tab", "").replace("-", " ").title()
        if isoutput in self.layout.output_tabs_functions.keys():
            cont = self.layout.output_tab(isoutput)
        else:
            tab_method = tab.replace("-", "_")
            cont = getattr(self.layout, tab_method)()
        return html.Div(cont)


    def store_parameters(self, n_clicks, **parameter_values):
        for l, d, params in self.layout.parameter_groups:
            accessor = getattr(self.project, d)
            stparams = {
                k: type(accessor[k])(parameter_values[k])
                for k in params
            }
            accessor(**stparams)
        return ["Saved!"]


    def run_model(self, set_progress, n_clicks, *parameter_values):
        import time
        from subprocess import PIPE, Popen

        # unpack parameters
        params = {pgn: v for pgn, v in
                  zip(self.layout.highlighted_parameters, parameter_values)}
        sim_start = dt.date(int(params[("config_parameters", "iyr")]), 1, 1)
        sim_end = dt.date(sim_start.year + int(params[("config_parameters", "nbyr")])-1, 12, 31)
        ndays = (sim_end - sim_start).days
        # set parameters in swim
        cfparams = self.project.config_parameters
        self.project.config_parameters(
            **{k: type(cfparams[k])(v) for (g, k), v in params.items()
            if g == "config_parameters"}
        )
        bsnparams = self.project.basin_parameters
        self.project.basin_parameters(
            **{k: type(bsnparams[k])(v) for (g, k), v in params.items()
            if g == "basin_parameters"}
        )


        swimcommand = [self.project.swim, self.project.projectdir+'/']
        process = Popen(swimcommand, stdout=PIPE, stderr=PIPE)

        prog = 0
        q = None
        while process.poll() is None:
            time.sleep(1)
            # reading discharge might fail due to incomplete lines
            try:
                q = self.project.station_daily_discharge
                prog = len(q)
            except Exception:
                pass
            qgraph = graphs.station_daily_discharge(q, sim_start, sim_end)
            set_progress((str(prog), str(ndays), "%1.0f%%" % (prog*100/ndays), qgraph))
        # make sure progress bar and graph are complete
        set_progress((str(ndays-1), str(ndays), "Saving...", qgraph))
        # run = self.project.save_run(tags=tags or "", notes=notes or "")
        qgraph = graphs.station_daily_discharge(self.project.station_daily_discharge, sim_start, sim_end)
        stdoutdata, stderrdata = process.communicate()
        return ["", qgraph]