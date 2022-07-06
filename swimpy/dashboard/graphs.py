import os.path as osp

from dash import html, dcc, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd

from swimpy.dashboard import layout, utils


def station_daily_discharge(station_daily_discharge, start, end):
    if station_daily_discharge is not None:
        q = (station_daily_discharge.stack()
            .reset_index()
            .set_axis(['time', "station", "discharge"], axis=1, inplace=False)
        )
        q.time = q.time.dt.to_timestamp()
        fig = px.line(q, x="time", y="discharge", color="station")
        graph = dcc.Graph(figure=fig, style=dict(height="50vh"))
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
    if hydrotope_daily_waterbalance is not None:
        hydv = (hydrotope_daily_waterbalance[["SURQ", "SUBQ", "PERC", "PLANT_ET", "SOIL_ET"]]
                .groupby(level=[2]).mean()).rename(columns=rename)
        hydv.index = hydv.index.to_timestamp()
        hydv = hydv.stack().reset_index().set_axis(['time', "variable", "mm"], axis=1)
        fig = px.line(hydv, x="time", y="mm", color="variable", title="Water balance")
        graph = dcc.Graph(figure=fig, style=dict(height="60vh"))
    else:
        graph = dcc.Graph(figure=px.line([]))

    graph.figure.update_xaxes(
        range=[str(start), str(end)],
        autorange = False
        )
    return graph


def plotly_hydrotopes_daily_waterbalance(project):
    hydwb = project.hydrotope_daily_waterbalance
    graph = hydrotopes_daily_waterbalance(
        hydwb,
        project.config_parameters.start_date,
        project.config_parameters.end_date,
    )
    hydv = (hydwb["ALAI"].groupby(level=[2]).mean())
    hydv.index = hydv.index.to_timestamp()
    fig = px.line(x=hydv.index, y=hydv, title="Leaf area index")
    laigraph = dcc.Graph(figure=fig, style=dict(height="50vh"))
    return dbc.Col([graph, laigraph])


def plotly_station_daily_and_regime_discharge(project):
    station_q = project.station_daily_discharge
    q = (station_q.stack()
        .reset_index()
        .set_axis(['time', "station", "discharge"], axis=1)
    )
    q.time = q.time.dt.to_timestamp()
    fig_daily = px.line(q, x="time", y="discharge", color="station", title="Daily discharge")
    fig_daily.update_layout(legend_y=-0.5, legend_x=0)
    graph_daily = dcc.Graph(figure=fig_daily, style=dict(width="55vw", height="50vh"))
    qdoy = station_q.groupby(station_q.index.dayofyear).mean()
    qdoyst = qdoy.stack().reset_index().set_axis(['DOY', "station", "discharge"], axis=1)
    fig_doy = px.line(qdoyst, x="DOY", y="discharge", color="station", title="Day of year mean")
    fig_doy.update_layout(showlegend=False)
    graph_doy = dcc.Graph(figure=fig_doy, style=dict(width="25vw", height="50vh"))
    return dbc.Row([graph_daily, graph_doy])


def plotly_station_daily_and_regime_discharge_component(project):
    q3 = project.catchment_daily_waterbalance[["SURQ", "SUBQ", "GWQ"]]
    q3.columns = "surface", "subsurface", "groundwater"
    q = q3.stack().reset_index().set_axis(['time', "discharge component", "discharge"], axis=1)
    q.time = q.time.dt.to_timestamp()
    fig_daily = px.line(q, x="time", y="discharge", color="discharge component", title="Discharge component")
    fig_daily.update_layout(legend_y=-0.5, legend_x=0)
    graph_daily = dcc.Graph(figure=fig_daily, style=dict(width="55vw", height="50vh"))

    qdoy = q3.groupby(q3.index.dayofyear).mean()
    qdoyst = qdoy.stack().reset_index().set_axis(['DOY', "discharge component", "discharge"], axis=1)
    fig_doy = px.line(qdoyst, x="DOY", y="discharge", color="discharge component", title="Day of year mean")
    fig_doy.update_layout(showlegend=False)
    graph_doy = dcc.Graph(figure=fig_doy, style=dict(width="25vw", height="50vh"))
    return dbc.Row([graph_daily, graph_doy])


def plotly_basin_daily_weather(project):
    bm = project.climate.inputdata.mean(axis=1, level=0)
    doy = bm.groupby(bm.index.dayofyear).mean()
    doy["day of year"] = doy.index
    vars = ["precipitation", "radiation", "humidity"]
    units = ["mm", "l/cm^2/day", "%"]
    bm['time'] = bm.index.to_timestamp()
    graphs = [
        dbc.Row([
            html.H3(v.title()),
            dcc.Graph(figure=px.line(x=bm.time, y=bm[v], labels={"y": u, "x": "time"}),  style=dict(width="55vw")),
            dcc.Graph(figure=px.line(x=doy.index, y=doy[v], labels={"y": u, "x": "day of year"}), style=dict(width="25vw")),
        ])
        for v, u in zip(vars, units)
    ]
    tvars = ["tmax", "tmean", "tmin"]
    graphs.insert(1, dbc.Row([
        html.H3("Temperature"),
        dcc.Graph(figure=px.line(bm, x="time", y=tvars, labels={"y": "dg C", "x": "time"}),  style=dict(width="55vw")),
        dcc.Graph(figure=px.line(doy, x="day of year", y=tvars, labels={"y": "dg C", "x": "time"}), style=dict(width="25vw")),
    ]))
    return dbc.Col(graphs)


def table_catchment_annual_waterbalance(project):
    df = project.catchment_annual_waterbalance.reset_index()
    df.loc["mean"] = df.mean()
    df.iloc[-1, 0] = "mean"
    df.iloc[:, 0] = df.iloc[:, 0].astype("string")
    df.columns = [c.title() for c in df.columns]
    table = dash_table.DataTable(df.round().to_dict("records"))
    return dbc.Row([html.H3("Annual catchment-wide water balance (mm)"), table])


def table_daily_discharge_performance(project):
    dq = project.station_daily_discharge
    perf = pd.concat([dq.NSE, dq.pbias], keys=["NSE", "% bias"], axis=1)
    perf.index.name = "station"
    table = dash_table.DataTable(perf.round(3).reset_index().to_dict("records"))
    return dbc.Row([html.H3("Discharge performance indicators", className="pt-4"), table])


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
    def map_func(project):
        label = gis_renames.get(hydattr, hydattr)
        values = getattr(project, hydattr).iloc[0]
        vmax = int(values.max()/10) * 10
        map = hydrotope_map(project, values, "jet_r", (0, vmax), "mm")
        div = dbc.Row([html.H3(label, className="pt-4"), map])
        return div
    return map_func


def plotly_reservoir(project):
    df = project.reservoir_output()
    df["time"] = df.index.to_timestamp()
    cont = dbc.Col([
        dbc.Row([
            html.H3("Inflow / outflow", className="pt-4"),
            dcc.Graph(figure=px.line(df, x="time", y=['Inflow_m3/s', 'Outflow_m3/s'], labels={"value": "m^3/s"}),  style=dict(width="80vw")),
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
