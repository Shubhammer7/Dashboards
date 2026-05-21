import plotly.express as px


def timeseries_plot(df, x, y, smooth=True):
    fig = px.line(df, x=x, y=y)
    if smooth:
        fig.update_traces(line_shape='spline')
    return fig


def barh(df, x, y, title=None):
    fig = px.bar(df, x=x, y=y, orientation='h', title=title)
    return fig
