"""
Plotly Dash callback functions with access to the swimpy project and the Dash layout.

- The input/output declarations to these functions/methods are provided as class attributes
  in the layout.Layout class, since they contain layout knowledge. 
- The number of arguments (expt. self) needs to match the output + state declarations in
  the above above class attributes.
"""
import os, os.path as osp
import base64
from dash import html, dcc
import dash_bootstrap_components as dbc
import datetime as dt
import io, zipfile
import tempfile
from flask import request

import dash_bootstrap_components as dbc

from . import graphs


class Callbacks:

    def __init__(self, project, layout, **overwrite):
        self.root_project = project
        self.layout = layout

    @property
    def project(self):
        username = request.authorization['username']
        return self.root_project.clone(username)

    def render_content(self, tab):
        isoutput = tab.replace("-tab", "").replace("-", " ").title()
        if isoutput in self.layout.output_tabs_functions.keys():
            cont = self.layout.output_tab(isoutput)
        else:
            tab_method = tab.replace("-", "_")
            cont = getattr(self.layout, tab_method)()
        return html.Div(cont)

    def render_output(self, n_clicks, tab, output_run, reference_run):
        tablab = tab.replace("-tab", "").replace("-", " ").title()
        return self.layout.render_output(tablab, output_run, reference_run)

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
        from swimpy.output import station_daily_discharge

        # unpack parameters
        params = {pgn: v for pgn, v in
                  zip(self.layout.highlighted_parameters, parameter_values)}
        # set climate (takes care of correct start/end year)
        climopt = params[("climate_parameters", "climate_input")]
        nbyropt = ("config_parameters", "nbyr")
        if climopt:
            self.project.climate.set_input(climopt)
            # make sure number of year is consistent with climate
            _, (cst, cend) = self.project.climate.options[climopt]
            if int(params[nbyropt]) - 1 > (cend - cst + 1):
                params[nbyropt] = (cend - cst + 1)
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

        # get number of days for progress bar
        sim_start = self.project.config_parameters.start_date
        sim_end = dt.date(sim_start.year + int(params[nbyropt])-1, 12, 31)
        nmax = (sim_end - sim_start).days
        freq = "d"
        # if more than 10 years, switch to monthly
        if nmax > 3650:
            freq = "m"
            nmax = nmax / 30.5 + 1
        prog = 0
        qfilepath = osp.join(self.project.projectdir, station_daily_discharge.path)
        qfile = open(qfilepath)
        header = None
        q = None
        while process.poll() is None:
            time.sleep(1)
            hydv = None
            # reading discharge might fail due to incomplete lines
            try:
                kw = {}
                if header:
                    kw = dict(header=None, names=header)
                qtimestep = self.project.station_daily_discharge.from_project(qfile, **kw)
                qtimestep = qtimestep if freq == "d" else qtimestep.resample(freq).mean()
                if header is None:
                    header = ["YEAR", "DAY", "observed"] + list(qtimestep.columns)
                    q = qtimestep
                elif qtimestep is not None:
                    q = q.append(qtimestep)
                prog = len(q)
            except Exception as e:
                print(e)
                pass
            qgraph = [
                graphs.station_daily_discharge(q, sim_start, sim_end),
                #graphs.hydrotopes_daily_waterbalance(hydv, sim_start, sim_end),
            ]
            set_progress((str(prog), str(nmax), "%1.0f%%" % (prog*100/nmax), qgraph))
        # make sure progress bar and graph are complete
        stdoutdata, stderrdata = process.communicate()
        print(stderrdata)
        qfile.close()
        # save output and report
        print("Saving output...")
        runkw = {k: v for (g, k), v in params.items() if g == "run_parameters" and bool(v)}
        if not runkw.get("notes", None):
            st, end = self.project.config_parameters.start_date, self.project.config_parameters.end_date
            runkw["notes"] = f"{climopt}, {st.year} - {end.year}"
        run = self.project.save_run(files=[], **runkw)
        nmax = len(self.project.save_run_files)
        for i, f in enumerate(self.project.save_run_files):
            set_progress((str(i+1), str(nmax), f"Saving {f}...", qgraph))
            try:
                df = self.project._attribute_or_function_result(f)
                self.project.save_file(run, f, df)
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                print(f"Failed saving {f}: {e}")
        # return full graph after run
        q = self.project.station_daily_discharge
        q = q if freq == "d" else q.resample(freq).mean()
        qgraph = [graphs.station_daily_discharge(q, sim_start, sim_end)]
        return ["", qgraph]

    def download_zipped_path(self, path):
        outpath = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with zipfile.ZipFile(outpath, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            if osp.isdir(path):
                zipdir(path, zip_file)
        return dcc.send_file(outpath.name)

    def download_input(self, n_clicks):
        pth = self.project.config_parameters["inputdir"]
        return self.download_zipped_path(pth)

    def download_output(self, n_clicks):
        pth = self.project.config_parameters["outputdir"]
        return self.download_zipped_path(pth)

    def upload_project(self, contents, filename, last_modified):
        inputdir = self.project.config_parameters["inputdir"]
        outputdir = self.project.projectdir
        expected = set([osp.relpath(osp.join(r, p), outputdir)
                        for r, d, fls in os.walk(inputdir) for p in fls])
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        msg = unzip(io.BytesIO(decoded), outputdir, expected)
        return dbc.Alert("Input updated!", dismissable=True)

    def reset_runs(self, n_clicks):
        nobs, nmods = self.project.browser.runs.all().delete()
        return dbc.Alert(f"All runs subset! ({nmods})", dismissable=True)


def unzip(file, destination, expected_files):
    expected_files = set(expected_files)
    with zipfile.ZipFile(file, 'r') as zip_ref:
        incomming = set(zip_ref.namelist())
        miss, redun = expected_files - incomming, incomming - expected_files
        if len(miss | redun):
            outp = ("Missing files: " + ", ".join(miss)) if miss else ""
            outp += (" Redundant files: " + ", ".join(redun)) if redun else ""
            return dbc.Alert(outp, color="warning", dismissable=True)
        print(f"Extracting {incomming} to {destination}")
        zip_ref.extractall(destination)
    return


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            relp = osp.relpath(osp.join(root, file), osp.join(path, '..'))
            ziph.write(osp.join(root, file), relp)