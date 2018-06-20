"""
SWIM related plotting functions and the generic plot_function decorator.

Standalone functions to create plots for SWIM input/output. They are used
throught the SWIMpy package but collected here to enable reuse.

All functions should accept an optional ax=None argument to plot to, i.e.
defaulting to the current axes (``ax = ax or plt.gca()``).

Project.method or Project.plugin.methods that implement plots should use the
``plot_function`` decorator to allow generic functionality.
"""
from __future__ import print_function, absolute_import
import sys
import tempfile
import functools
import datetime as dt

import numpy as np
import pandas as pd
from modelmanager.settings import FunctionInfo
import matplotlib as mpl
# needed to use matplotlib in django browser
if len(sys.argv) > 1 and sys.argv[1] == 'browser':
    mpl.use('Agg')
import matplotlib.pyplot as plt


def save(output, figure=None, **savekwargs):
    """Convenience function to set figure size and save a matplotlib figure.

    Arguements:
    -----------
    output : str
        Path to save figure to. Extension determines format.
    figure : matplotlib.Figure object, optional
        Defaults to current figure.
    **savekwargs :
        Any keyword argument parsed to figure.savefig() method. Special keys:
        ``size`` : len 2 tuple
            Size in mm.

    Returns: None
    """
    figure = figure or plt.gcf()
    assert type(output) == str, 'output %r must be string path.'
    size = savekwargs.pop('size', None)
    if size:
        assert len(size) == 2, 'size must be (width, height) not %r' % size
        mmpi = 25.4
        figure.set_size_inches(size[0]/mmpi, size[1]/mmpi)  # (width, hight)
    figure.savefig(output, **savekwargs)
    return


def plot_waterbalance(series, ax=None, **barkwargs):
    """Bar plot of water balance terms.

    Arguments:
    ----------
    df : pd.Series
        Values to plot. Index will be used as x labels.
    ax : plt.Axes, optional
        An axes to plot to. If None given, the current axes are used.
    **barkwargs :
        plt.bar keyword arguments.

    Returns: bars
    """
    ax = plt.gca()
    bars = series.plot.bar(ax=ax, **barkwargs)
    ax.set_ylabel('mm per year')
    ax.set_title('Catchment mean water balance')
    return bars


def plot_temperature_range(series, ax=None, minmax=[], **linekwargs):
    """Plot temperature with optional min-max range."""
    assert len(minmax) in [0, 2]
    ax = ax or plt.gca()
    if minmax:
        kw = dict(alpha=0.3, color='k')
        mmfill = ax.fill_between(_index_to_timestamp(series.index), minmax[0],
                                 minmax[1], **kw)
    line = ax.plot(_index_to_timestamp(series.index), series, **linekwargs)
    ax.set_ylabel('Temperature [C]')
    ax.set_xlabel('Time')
    return (line, mmfill) if minmax else line


def plot_precipitation_bars(series, ax=None, **barkwargs):
    """Plot precipitation as bars."""
    ax = ax or plt.gca()
    if hasattr(series.index, 'to_timestamp'):
        freqstr = series.index.freqstr.split('-')[0][-1].lower()  # last letter
        width = {'a': 365, 'm': series.index.days_in_month, 'd': 1}
        barkwargs.setdefault('width', width[freqstr]*0.8)
    bars = ax.bar(_index_to_timestamp(series.index), series, **barkwargs)
    ax.set_ylabel('Precipitation [mm]')
    ax.set_xlabel('Time')
    return bars


def plot_discharge(series, ax=None, **linekwargs):
    """Plot several discharge lines."""
    ax = ax or plt.gca()
    lines = ax.plot(_index_to_timestamp(series.index), series, **linekwargs)
    ax.set_ylabel('Discharge [m$^3$s$^{-1}$]')
    ax.set_xlabel('Time')
    return lines


def plot_flow_duration_polar(series, axes=None, percentilestep=10,
                             freq='m', colormap='jet_r', **barkw):
    """Bins the values in series into 100/percentilestep steps and displays the
    relative frequency per month or day-of-year (freq= m|f) on a polar bar
    chart of the year. See in action and more docs in:
    `output.station_daily_discharge.plot_flow_duration_polar`
    """
    assert percentilestep <= 50
    axes = axes or plt.gca()
    # exchange axes for polar axes
    apo = axes.get_position()
    axes.set_axis_off()
    axes = plt.gcf().add_axes(apo, projection='polar')

    ssorted = series.dropna().sort_values()
    n = len(ssorted)
    dom = 'dayofyear' if freq.lower() == 'd' else 'month'
    ib = 0
    count = {}
    for b in range(percentilestep, 100+1, percentilestep):
        # get percentile bin
        iib = min(int(round(n*b/100.)), n)
        bin = ssorted.iloc[ib:iib]
        # count values in percentile bin
        count[b] = bin.groupby(getattr(bin.index, dom)).count()/float(n)
        ib = iib

    ntb = 365 if freq.lower() == 'd' else 12
    theta = np.arange(ntb) * 2 * np.pi / ntb
    cm = plt.get_cmap(colormap)
    countdf = pd.DataFrame(count).loc[:ntb]
    countdf /= countdf.max().max()
    countdf.fillna(0, inplace=True)
    for b, col in countdf.items():
        bars = axes.bar(x=theta, height=[percentilestep]*len(theta),
                        width=2*np.pi/ntb, bottom=b-percentilestep,
                        color=cm(col), edgecolor='none')
    axes.set_theta_zero_location('N')
    axes.set_theta_direction(-1)
    axes.set_rmin(0)
    axes.set_rmax(100)
    axes.grid(False)
    month_names = [dt.date(2000, i, 1).strftime('%B') for i in range(1, 13)]
    tcks, tcklbls = plt.xticks(np.arange(12)*2*np.pi/12, month_names)
    rots = (list(range(0, -91, -30)) + list(range(60, -61, -30)) +
            list(range(90, 30-1, -30)))
    for l, r in zip(tcklbls, rots):
        l.set_rotation(r)
    axes.set_yticks([50])
    axes.set_yticklabels(['50%'])
    axes.grid(True, axis='y')
    return axes


def _index_to_timestamp(index):
    """Convert a pandas index to timestamps if needed.
    Needed to parse pandas PeriodIndex to pyplot plotting functions."""
    return index.to_timestamp() if hasattr(index, 'to_timestamp') else index


def plot_function(function):
    """Decorator for the PlotFunction class.

    This factory function is required to return a function rather than an
    object if PlotFunction was used as a decorator alone.
    """
    pf = PlotFunction(function)

    @functools.wraps(function)
    def f(*args, **kwargs):
        return pf(*args, **kwargs)
    # add signiture to beginning of docstrign if PY2
    if sys.version_info < (3, 0):
        sig = '%s(%s)\n' % (pf.finfo.name, pf.finfo.signiture)
        f.__doc__ = sig + pf.finfo.doc
    # add generic docs
    function.__doc__ = (pf.finfo.doc or '') + PlotFunction.additional_docs
    # attach original function
    f.decorated_function = function
    return f


class PlotFunction(object):
    """A a class that enforces and handles generic plot function tasks.

    To be used in plot_function decorator.

    - enforces name starting with 'plot'.
    - enforces output=None and ax=None arugments.
    - enforces to accept **kwargs.
    - enforces the method instance (first function argement) to either be a
      project or have a project attribute
    - reads savefig_defaults from project
    - allows saving figure to file with output argument that may either be
      string path or a dict with kwargs to save.
    - displays interactive plot if executed from commandline.
    - saves current figure to a temp path when executed in browser API.
    - allows running function with a run instance if the function has a run
      argument. The argument input is normalised (see additional_docs).
    """
    additional_docs = """

Plot function arguments:
------------------------
ax : <matplotlib.Axes>, optional
    Axes to plot to. Default is the current axes if None.
output : str path | dict
    Path to writeout or dict of keywords to parse to save_or_show.
runs : Run | runID | iterable of Run/runID | QuerySet, optional
    Show plot for runs if they have the same method or plugin.method. The runs
    input is transformed to (run QuerySet, index) to enable per run styling.
    """

    def __init__(self, function):
        self.decorated_function = function
        # enforce arugments
        self.finfo = FunctionInfo(function)
        oargs = dict(zip(self.finfo.optional_arguments, self.finfo.defaults))
        errmsg = self.finfo.name + ' has no optional argument "%s=None".'
        for a in ['output', 'ax']:
            assert a in oargs and oargs[a] is None, errmsg % a
        errmsg = self.finfo.name + ' should start with "plot".'
        assert self.finfo.name.startswith('plot'), errmsg
        assert self.finfo.kwargs, self.finfo.name+' must accept **kwargs.'
        # attributes assigned during call
        callattr = ('project instance args kwargs ax figure result savekwargs '
                    'output runs')
        for a in callattr.split():
            setattr(self, a, None)
        return

    def _infer_project(self):
        from .project import Project
        self.instance = self.args[0]  # assumes method
        if isinstance(self.instance, Project):
            self.project = self.instance
        elif hasattr(self.instance, 'project'):
            self.project = self.instance.project
        else:
            em = '%s is not a Project instance or has a project attribute.'
            raise AttributeError(em % self.instance)
        return

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.runs = kwargs.get('runs')
        self.output = kwargs.get('output')
        self.ax = kwargs.get('ax', plt.gca()) or plt.gca()
        self.figure = self.ax.get_figure() if self.ax else plt.gcf()
        self._infer_project()

        if self.runs:
            # transform runs to QuerySet
            self.runs = self.project.browser.runs.get_runs(self.runs)
            self._plot_runs()
        else:
            self.result = self.decorated_function(*args, **kwargs)

        if self.output:
            self._get_savekwargs()
            save(self.output, self.figure, **self.savekwargs)
        # display if from commandline or browser api
        elif sys.argv[0].endswith('swimpy'):
            self._display_figure()
        return self.result

    def _plot_runs(self):
        ispi = self.instance.__class__ != self.project.__class__
        piname = self.instance.__class__.__name__  # project if not plugin
        self.result = [self.result]
        for i, r in enumerate(self.runs):
            try:
                piinstance = r
                if ispi:  # if project.plugin
                    piinstance = getattr(r, piname)
                pmeth = getattr(piinstance, self.finfo.name)
            except AttributeError:
                m = self.finfo.name if ispi else piname+'.'+self.finfo.name
                print('%s doesnt have a %s method.' % (r, m))
                continue
            rkw = self.kwargs.copy()
            rkw['runs'] = (self.runs, i)
            rkw.setdefault('label', str(r))
            # call method with different instance as first argument as
            # decorated_function is unbound
            rre = pmeth.decorated_function(piinstance, *self.args[1:], **rkw)
            self.result.append(rre)
        # make sure a legend is shown if not already
        if self.ax.get_legend() is None:
            self.ax.legend()
        return

    def _get_savekwargs(self):
        # unpack savekwargs
        self.savekwargs = {}
        self.savekwargs.update(self.project.save_figure_defaults)
        if type(self.output) is dict:
            op = self.output.pop('output', None)
            self.savekwargs.update(self.output)
            self.output = op
        return

    def _display_figure(self):
        # in Django API
        if len(sys.argv) > 1 and sys.argv[1] == 'browser':
            imgpath = tempfile.mkstemp()[1] + '.png'
            save(imgpath, self.figure, **self.savekwargs)
            self.figure.clear()
            self.result = imgpath
        else:  # in CLI
            plt.show(block=True)
        return
