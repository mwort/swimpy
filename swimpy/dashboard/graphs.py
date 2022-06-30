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
    return [graph]


def plotly_station_daily_discharge(project):
    q = project.station_daily_discharge
    graph = station_daily_discharge(
        q,
        project.config_parameters.start_date,
        project.config_parameters.end_date,
    )
    return graph[0]


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
    vars = bm.columns
    bm['time'] = bm.index.to_timestamp()
    graphs = html.Div([
        dbc.Row([
            html.H3(v.title()),
            dcc.Graph(figure=px.line(x=bm.time, y=bm[v]),  style=dict(width="55vw")),
            dcc.Graph(figure=px.line(x=doy.index, y=doy[v]), style=dict(width="25vw")),
        ])
        for v in vars
    ])
    return graphs


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
    return dbc.Row([html.H3("Discharge performance indicators"), table])


def hydrotope_map(project, hydrotope_values):
    import dash_leaflet as dl
    import hashlib

    # check if already produced
    md5 = hashlib.md5(hydrotope_values.to_string().encode()).hexdigest()
    tmppth = osp.join(project.browser.settings.tmpfilesdir, md5+"_hydrotope_map.png")
    if not osp.exists(tmppth):
        project.hydrotopes.to_image(hydrotope_values, tmppth)

    # assemble map
    image_bounds = project.hydrotopes.array_latlon_bounds
    map = dl.Map([
        dl.ImageOverlay(opacity=0.66, url=utils.local_img(tmppth, imgtag=False), bounds=image_bounds), dl.TileLayer()],
        bounds=image_bounds,
        style={'width': '50vw', 'height': '50vh', 'margin': "auto", "display": "block"})
    return map


def map_hydrotope_means(hydattr):
    def map_func(project):
        values = getattr(project, hydattr).iloc[0]
        div = html.Div([html.H3(hydattr), hydrotope_map(project, values)])
        return div
    return map_func