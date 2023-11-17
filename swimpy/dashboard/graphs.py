import os.path as osp

from dash import html, dcc, dash_table
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from swimpy.dashboard import layout, utils


def station_daily_discharge(station_daily_discharge, start, end):
    if station_daily_discharge is not None:
        q = (station_daily_discharge.stack()
            .reset_index()
            .set_axis(['time', "station", "discharge"], axis=1, inplace=False)
        )
        q.time = q.time.dt.to_timestamp()
        fig = px.line(q, x="time", y="discharge", color="station")
        fig.update_layout(xaxis_title="Time [days]", yaxis_title="Discharge [m&#179;/s]")
        graph = dcc.Graph(figure=fig, style=dict(height="70vh"))
    else:
        graph = dcc.Graph(figure=px.line([]))

    graph.figure.update_xaxes(
        range=[str(start), str(end)],
        autorange = False
        )
    return graph


def plotly_station_daily_discharge(project):
    q = project.station_daily_discharge
    graph = station_daily_discharge(
        q,
        project.config_parameters.start_date,
        project.config_parameters.end_date,
    )
    return graph


def hydrotopes_daily_waterbalance(hydrotope_daily_waterbalance, start, end):
    rename = {"SURQ": "surface runoff", "SUBQ": "subsurface runoff",
              "PERC": "percolation", "PLANT_ET": "plant transpiration", "SOIL_ET": "soil evapotranspiration"}

    hydv = (hydrotope_daily_waterbalance[["SURQ", "SUBQ", "PERC", "PLANT_ET", "SOIL_ET"]]
            .groupby(level=[2]).mean()).rename(columns=rename)
    hydv.index = hydv.index.to_timestamp()
    doy = hydv.groupby(hydv.index.dayofyear).mean().loc[:365]
    hydv, doy = (df.stack().reset_index().set_axis(['time', "variable", "mm"], axis=1) for df in (hydv, doy))

    daily = px.line(hydv, x="time", y="mm", color="variable", title="Daily")
    daily.update_layout(legend_title="", yaxis_title="mm/d")
    dailygraph = dcc.Graph(figure=daily, style=dict(height="60vh", width="60vw"))
    dy = px.line(doy, x="time", y="mm", color="variable", title="Day of year mean")
    dy.update_layout(showlegend=False, xaxis_title="Day of year", yaxis_title="mm/d")
    doygraph = dcc.Graph(figure=dy, style=dict(height="60vh", width="35vw"))

    return dbc.Row([dailygraph, doygraph])


def plotly_hydrotopes_daily_waterbalance(project, run=None, reference=None):
    hydwb = project.hydrotope_daily_waterbalance
    graph = hydrotopes_daily_waterbalance(
        hydwb,
        project.config_parameters.start_date,
        project.config_parameters.end_date,
    )
    hydv = (hydwb["ALAI"].groupby(level=[2]).mean())
    hydv.index = hydv.index.to_timestamp()
    doylai = hydv.groupby(hydv.index.dayofyear).mean().loc[:365]

    fig = px.line(x=hydv.index, y=hydv, title="Leaf area index")
    fig.update_layout(xaxis_title="Time [days]", yaxis_title="LAI [~]")
    laigraphdaily = dcc.Graph(figure=fig, style=dict(height="50vh", width="60vw"))
    fig = px.line(x=doylai.index, y=doylai)
    fig.update_layout(xaxis_title="Day of year", yaxis_title="LAI [~]")
    laigraphdoy = dcc.Graph(figure=fig, style=dict(height="50vh", width="35vw"))
    laigraph = dbc.Row([laigraphdaily, laigraphdoy])

    return dbc.Col([graph, laigraph])


def plotly_station_daily_and_regime_discharge(project, run=None, reference=None):
    station_q = run.station_daily_discharge
    levels = ["color", "line_dash", "x", "y"]
    level_cols = ["run", "station", 'time', "discharge"]
    if reference:
        station_q = pd.concat([station_q, reference.station_daily_discharge],
                              keys=[str(run), str(reference)],
                              names=["run", "station"], axis=1)
    else:
        station_q.columns = pd.MultiIndex.from_tuples([(str(run), s) for s in station_q.columns])
    # check if observation
    obs, _ = run.station_daily_discharge.obs_sim_overlap()
    if len(obs):
        for st, dat in obs.items():
            station_q.loc[dat.index, ("observations", st)] = dat
    q = (station_q.stack([0, 1]).swaplevel(1, 0).swaplevel(2, 1)
        .reset_index().set_axis(level_cols, axis=1)
    )
    q.time = q.time.dt.to_timestamp()
    q["DOY"] = q.time.dt.day_of_year
    qdoy = q.groupby(list({"DOY"} | set(level_cols) - {"time", "discharge"})).mean().reset_index()
    q["year"] = q.time.dt.year
    qan = q.groupby(list({"year"} | set(level_cols) - {"time", "discharge"})).mean().reset_index()

    linekw = dict(zip(levels, level_cols))
    layoutkw = dict(legend_title="", margin=dict(r=20, l=20, t=40), yaxis_title="Discharge [m&#179;/s]")
    fig_daily = px.line(q, title="Daily discharge", **linekw)
    fig_daily.update_layout(**layoutkw)
    graph_daily = dcc.Graph(figure=fig_daily, style=dict(width="90vw", height="60vh"))

    linekw["x"] = "DOY"
    fig_doy = px.line(qdoy, title="Day of year mean", **linekw)
    fig_doy.update_layout(showlegend=False, **layoutkw)
    graph_doy = dcc.Graph(figure=fig_doy, style=dict(width="40vw", height="60vh"))

    linekw["x"] = "year"
    fig_an = px.line(qan, title="Annual mean discharge", **linekw)
    fig_an.update_layout(showlegend=False, **layoutkw)
    graph_an = dcc.Graph(figure=fig_an, style=dict(width="55vw", height="60vh"))
    return [dbc.Row(graph_daily), dbc.Row([graph_an, graph_doy])]


def plotly_station_daily_and_regime_discharge_component(project, run=None, reference=None):
    q3 = project.catchment_daily_waterbalance[["SURQ", "SUBQ", "GWQ"]]
    q3.columns = "surface", "subsurface", "groundwater"
    q = q3.stack().reset_index().set_axis(['time', "discharge component", "discharge"], axis=1)
    q.time = q.time.dt.to_timestamp()
    qdoy = q3.groupby(q3.index.dayofyear).mean()
    qdoyst = qdoy.stack().reset_index().set_axis(['DOY', "discharge component", "discharge"], axis=1)

    layoutkw = dict(legend_title="", margin=dict(r=20, l=20, t=100), yaxis_title="Discharge [m&#179;/s]")

    fig_daily = px.line(q, x="time", y="discharge", color="discharge component", title="Discharge component")
    fig_daily.update_layout(**layoutkw)
    graph_daily = dcc.Graph(figure=fig_daily, style=dict(width="55vw", height="50vh"))

    fig_doy = px.line(qdoyst, x="DOY", y="discharge", color="discharge component", title="Day of year mean")
    fig_doy.update_layout(showlegend=False, **layoutkw)
    graph_doy = dcc.Graph(figure=fig_doy, style=dict(width="35vw", height="50vh"))
    return dbc.Row([graph_daily, graph_doy])


def plotly_basin_daily_weather(project, run, reference=None):
    bm = run.catchment_daily_temperature_precipitation
    if reference and run != reference:
        bm = pd.concat([bm, reference.catchment_daily_temperature_precipitation],
                        keys=[str(run), str(reference)],
                        names=["run", "variable"], axis=1).swaplevel(axis=1)
    else:
        bm.columns = pd.MultiIndex.from_tuples([(s, str(run)) for s in bm.columns])
    doy = bm[["precipitation", "tmean"]].groupby(bm.index.dayofyear).mean().loc[:365]
    annual = bm[["precipitation", "tmean"]].resample("a").mean()
    annual["precipitation"] = bm["precipitation"].resample("a").agg( pd.Series.sum, skipna=False)
    dayix = bm.index.to_timestamp()

    v = "precipitation"
    barkw = dict(bargap=0., barmode="group")
    daily_ax = px.bar(bm[v], x=dayix, y=list(bm[v].columns))
    daily_ax.update_layout(showlegend=bool(reference), yaxis_title="Precipitation [mm/d]",
                           xaxis_title="Time [days]", title="Daily", legend_title="Runs", **barkw)

    doy_ax = px.bar(doy[v], x=doy.index.values, y=list(doy[v].columns))
    doy_ax.update_layout(showlegend=False, yaxis_title="Precipitation [mm/d]",
                         title="Day of year mean", xaxis_title="Time [day of year]", **barkw)

    annual_ax = px.bar(annual[v], x=annual.index.year, y=list(annual[v].columns))
    annual_ax.update_layout(showlegend=False, yaxis_title="Precipitation [mm/a]", xaxis_title="Time [year]",
                            title="Annual sum", **barkw)

    graphs = [
        dbc.Row([
            html.H3(v.title()),
            dcc.Graph(figure=daily_ax, style=dict(width="100vw")),
        ]),
        dbc.Row([
            dcc.Graph(figure=doy_ax, style=dict(width="45vw")),
            dcc.Graph(figure=annual_ax, style=dict(width="45vw")),
        ]),
    ]

    tvars = "tmean"
    tmpkw = dict(yaxis_title="Temperature [&deg;C]")

    daily_ax = px.line(bm[tvars], x=dayix, y=list(bm[tvars].columns))
    daily_ax.update_layout(showlegend=bool(reference), xaxis_title="Time [days]", title="Daily",
                           legend_title="Runs", **tmpkw)

    doy_ax = px.line(doy[tvars], x=doy.index, y=list(doy[tvars].columns))
    doy_ax.update_layout(showlegend=False, title="Day of year mean", xaxis_title="Time [day of year]", **tmpkw)

    annual_ax = px.line(annual[tvars], x=annual.index.year, y=list(annual[tvars].columns))
    annual_ax.update_layout(showlegend=False, xaxis_title="Time [year]", title="Annual mean", **tmpkw)

    graphs += [
        dbc.Row([
            html.H3("Temperature"),
            dcc.Graph(figure=daily_ax, style=dict(width="100vw")),
        ]),
        dbc.Row([
            dcc.Graph(figure=doy_ax, style=dict(width="45vw")),
            dcc.Graph(figure=annual_ax, style=dict(width="45vw")),
        ])
    ]
    return dbc.Col(graphs)


def table_catchment_annual_waterbalance(project, run=None, reference=None):
    rundat = run.catchment_annual_waterbalance
    df = rundat.reset_index()
    df.loc["mean"] = df.mean()
    df.iloc[-1, 0] = "mean"
    df.iloc[:, 0] = df.iloc[:, 0].astype("string")
    df.columns = [c.title() for c in df.columns]
    table = dash_table.DataTable(df.round().to_dict("records"))
    title = f"{run}: Annual catchment-wide water balance (mm)"
    rows = dbc.Row([html.H3(title), table], class_name="py-4")
    if reference:
        ro = table_catchment_annual_waterbalance(project, reference)
        # difference
        rdf = reference.catchment_annual_waterbalance.mean()
        dif = ((rundat.mean() - rdf) * 100 / rdf).to_frame().T
        dif.columns = [c.title() for c in dif.columns]
        table = dash_table.DataTable(dif.round(2).to_dict("records"))
        title = f"Change {run} - {reference} (%)"
        difrow = dbc.Row([html.H3(title), table], class_name="py-4")
        rows = dbc.Row([difrow, rows, ro])
    return rows


def table_daily_discharge_performance(project, run, reference=None):
    dq = run.station_daily_discharge
    obs, _ = dq.obs_sim_overlap()
    if len(obs) == 0:
        return
    overl = obs.apply(lambda o: ' - '.join(o.dropna().index[[0, -1]].astype(str)))
    perf = pd.concat([dq.NSE, dq.pbias, overl], keys=["NSE", "% bias", "time"], axis=1)
    perf.index.name = "station"
    table = dash_table.DataTable(perf.round(3).reset_index().to_dict("records"))
    return dbc.Row([html.H5("Discharge performance indicators", className="pt-4"), table])


def map_colorbar(cmap_name, path, vminmax, unit):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(1.1, 5))
    fig.subplots_adjust(right=0.35)

    norm = mpl.colors.Normalize(vmin=vminmax[0], vmax=vminmax[1])
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=getattr(mpl.cm, cmap_name))
    fig.colorbar(cmap, cax=ax, orientation='vertical', label=unit, extend="max")
    fig.savefig(path)
    return cmap


def hydrotope_map(project, hydrotope_values, cmap_name, vminmax, unit):
    import dash_leaflet as dl
    import hashlib
    from matplotlib import colors, pyplot

    cbpth = osp.join(project.browser.settings.tmpfilesdir,
                     f"colorbar_{cmap_name}_{vminmax[0]}_{vminmax[1]}.png")
    if not osp.exists(cbpth):
        map_colorbar(cmap_name, cbpth, vminmax, unit)

    # check if already produced
    md5 = hashlib.md5(hydrotope_values.to_string().encode()).hexdigest()
    tmppth = osp.join(project.browser.settings.tmpfilesdir, md5+"_hydrotope_map.png")
    if not osp.exists(tmppth):
        project.hydrotopes.to_image(hydrotope_values, tmppth, vminmax, cmap=cmap_name)

    # assemble map
    image_bounds = project.hydrotopes.array_latlon_bounds
    map = dl.Map([
        dl.ImageOverlay(opacity=0.5, url=utils.local_img(tmppth, imgtag=False), bounds=image_bounds), dl.TileLayer()],
        bounds=image_bounds,
        style={'width': '50vw', 'height': '60vh', 'margin': "auto", "display": "block"})
    return dbc.Row([dbc.Col([map], className="col-sm-8"), dbc.Col([utils.local_img(cbpth)], className="col-sm-2")])


gis_renames = {
    "hydrotope_evapmean_gis": "Evapotranspiration",
    "hydrotope_gwrmean_gis": "Groundwater recharge",
    "hydrotope_pcpmean_gis": "Precipitation",
}

def map_hydrotope_means(hydattr):
    def map_func(project, run, reference=None):
        label = gis_renames.get(hydattr, hydattr)
        values = getattr(run, hydattr).iloc[0]
        unit = "mm"
        if reference:
            refvalues = getattr(reference, hydattr).iloc[0]
            values = (values - refvalues) * 100 / refvalues
            unit = "%"
        vmax = (int(values[np.isfinite(values)].max()/10) + 1) * 10
        vmin = int(values[np.isfinite(values)].min()/10) * 10
        print(f"Creating map {hydattr}")
        map = hydrotope_map(project, values, "jet_r", (vmin, vmax), unit)
        div = dbc.Row([html.H3(label, className="pt-4"), map])
        return div
    return map_func


def plotly_reservoir(project, run=None, reference=None):
    df = project.reservoir_output()
    df["time"] = df.index.to_timestamp()
    cont = dbc.Col([
        dbc.Row([
            html.H3("Inflow / outflow", className="pt-4"),
            dcc.Graph(figure=px.line(df, x="time", y=['Inflow_m3/s', 'Outflow_m3/s'], labels={"value": "m&#179;/s"}),  style=dict(width="80vw")),
        ]),
        dbc.Row([
            html.H3("Water level", className="pt-4"),
            dcc.Graph(figure=px.line(df, x="time", y='WaterLevel_masl', labels={'WaterLevel_masl': "m asl."}),  style=dict(width="80vw")),
        ]),
        dbc.Row([
            html.H3("Electricity production", className="pt-4"),
            dcc.Graph(figure=px.line(df, x="time", y='Energy_MW', labels={'Energy_MW': "MW"}),  style=dict(width="80vw")),
        ]),
    ])
    return cont
