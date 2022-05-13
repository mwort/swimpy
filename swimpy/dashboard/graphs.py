from dash import html, dcc
import plotly.express as px

from swimpy.dashboard import layout


def station_daily_discharge(station_daily_discharge, start, end):
    if station_daily_discharge is not None:
        q = (station_daily_discharge.stack()
            .reset_index()
            .set_axis(['time', "station", "discharge"], axis=1, inplace=False)
        )
        q.time = q.time.dt.to_timestamp()
        fig = px.line(q, x="time", y="discharge", color="station")
        graph = dcc.Graph(figure=fig)
    else:
        graph = dcc.Graph(figure=px.line([]))
    graph.figure.update_xaxes(
        range=[str(start), str(end)],
        autorange = False
        )
    return [graph]


def basin_daily_weather(project):
    bm = project.climate.inputdata.mean(axis=1, level=0)
    vars = bm.columns
    bm['time'] = bm.index.to_timestamp()
    graphs = html.Div([
        html.Div([
            html.H3(v.title()),
            dcc.Graph(figure=px.line(x=bm.time, y=bm[v]))
        ])
        for v in vars
    ])
    return graphs
