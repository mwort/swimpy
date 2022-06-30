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
        return html.Div(self.layout.tabs_content[tab])

    def run_model(self, set_progress, n_clicks, years, year):
        import time
        from subprocess import PIPE, Popen
        sim_start = dt.date(int(year), 1, 1)
        sim_end = dt.date(int(year)+int(years)-1, 12, 31)
        ndays = (sim_end - sim_start).days

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