"""
SWIM related plotting functions and the generic plot_function decorator.

Standalone functions to create plots for SWIM input/output. They are used
throught the SWIMpy package but collected here to enable reuse.

All functions should accept an optional ax=None argument to plot to. This
argument will always be converted to a valid axes (i.e. plt.gca() if None).

Project.method or Project.plugin.methods that implement plots should use the
``plot_function`` decorator to allow generic functionality.
"""
from __future__ import print_function, absolute_import
import sys
import tempfile
import functools
import datetime as dt
import warnings

import numpy as np
import pandas as pd
from modelmanager.settings import FunctionInfo, parse_settings
import matplotlib as mpl
# needed to use matplotlib in django browser
if len(sys.argv) > 1 and sys.argv[1] == 'browser':
    mpl.use('Agg')
import matplotlib.pyplot as plt


def save(output, figure=None, tight_layout=True, **savekwargs):
    """Convenience function to set figure size and save a matplotlib figure.

    Arguments
    ---------
    output : str
        Path to save figure to. Extension determines format.
    figure : matplotlib.Figure object, optional
        Defaults to current figure.
    tight_layout : bool
        Apply ``pyplot.tight_layout`` to figure reducing figure whitespace.
    **savekwargs :
        Any keyword argument parsed to ``figure.savefig()`` method.
        Special keys:
        ``size`` : len 2 tuple, size in mm.
    """
    figure = figure or plt.gcf()
    assert type(output) == str, 'output %r must be string path.'
    size = savekwargs.pop('size', None)
    if size:
        assert len(size) == 2, 'size must be (width, height) not %r' % size
        mmpi = 25.4
        figure.set_size_inches(size[0]/mmpi, size[1]/mmpi)  # (width, hight)
    if tight_layout:
        # tight_layout doesnt work with irregular grid plots, lets try
        try:
            figure.tight_layout()
        except RuntimeError:
            warnings.warn('Figure doesnt allow tight layout.')
    figure.savefig(output, **savekwargs)
    return


def plot_waterbalance(series, ax=None, **barkwargs):
    """Bar plot of water balance terms.

    Arguments
    ---------
    df : pd.Series
        Values to plot. Index will be used as x labels.
    ax : plt.Axes, optional
        An axes to plot to. If None given, the current axes are used.
    **barkwargs :
        plt.bar keyword arguments.

    Returns
    -------
    bars
    """
    ax = ax or plt.gca()
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
    :meth:`swimpy.output.station_daily_discharge.plot_flow_duration_polar`
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


def plot_objective_scatter(performances, selected=None, ax=None, **scatterkw):
    '''Plot scatter against all objectives combinations in a stepped subplot.

    Arguments
    ---------
    performances : pd.DataFrame
        DataFrame with performance values.
    selected : dict-like
        Highlight one selected point.
    '''
    objectives = performances.columns
    # calculate limits
    nticks = 5
    margin = 0.1  # fraction of median
    stats = performances.describe()
    rng = (stats.ix['max'] - stats.ix['min']) * margin
    limits = {'max': stats.ix['max'] + rng,
              'min': stats.ix['min'] - rng}

    naxes = len(objectives) - 1
    if ax:
        f = ax.get_figure()
        axs = f.get_axes()
        if len(axs) == naxes**2:
            ax = np.array(axs).reshape(naxes, naxes)
        else:
            f.clear()
            ax = None
    else:
        f = plt.figure()
    if ax is None:
        ax = f.subplots(naxes, naxes, squeeze=False)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, n in enumerate(objectives[1:]):  # row
        for ii, nn in enumerate(objectives[:-1]):  # column
            if ii <= i:
                ax[i][ii].scatter(
                    performances[nn], performances[n], **scatterkw)
                if selected is not None:
                    ax[i][ii].scatter(selected[nn], selected[n], c='r')
                # axis adjustments
                xticks = mpl.ticker.MaxNLocator(nbins=nticks, prune='upper')
                ax[i][ii].xaxis.set_major_locator(xticks)
                yticks = mpl.ticker.MaxNLocator(nbins=nticks, prune='upper')
                ax[i][ii].yaxis.set_major_locator(yticks)
                ax[i][ii].set_ylim(limits['min'][n], limits['max'][n])
                ax[i][ii].set_xlim(limits['min'][nn], limits['max'][nn])
            else:  # remove unused axes
                ax[i][ii].set_frame_on(False)
                ax[i][ii].set_xticks([])
                ax[i][ii].set_yticks([])
            # labels
            if i == naxes - 1:
                ax[i][ii].set_xlabel(nn)
            else:
                ax[i][ii].set_xticklabels([])

            if ii == 0:
                ax[i][0].set_ylabel(n)
            else:
                ax[i][ii].set_yticklabels([])
    return ax


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
    docs = (pf.finfo.doc or '') + PlotFunction.ax_output_docs
    if 'runs' in pf.finfo.optional_arguments:
        docs += PlotFunction.runs_docs
    function.__doc__ = docs
    # attach original function
    f.decorated_function = function
    return f


class PlotFunction(object):
    """A a class that enforces and handles generic plot function tasks.

    To be used in plot_function decorator.

    - enforces name starting with 'plot'.
    - enforces ax=None arugment and ensures a valid axes is always parsed.
    - enforces to accept ``**kwargs``.
    - enforces the method instance (first function argement) to either be a
      project or have a project attribute
    - reads savefig_defaults from project
    - enforces output=None argument and allows saving of figure to file with
      that may either be string path or a dict with kwargs to save.
    - displays interactive plot if executed from commandline.
    - saves current figure to a temp path when executed in browser API.
    - allows running function with a run instance if the function has a run
      argument. The argument input is normalised (see additional_docs).
    """
    ax_output_docs = """

Plot function arguments:
------------------------
ax : <matplotlib.Axes>, optional
    Axes to plot to. Default is the current axes if None.
output : str path | dict
    Path to writeout or dict of keywords to parse to save_or_show."""
    runs_docs = """
runs : Run | runID | iterable of Run/runID | QuerySet | (str), optional
    Show plot for runs if they have the same method or plugin.method. If a
    string is parsed, the current project will also be plot with the string as
    label. The runs argument is transformed to (run QuerySet, index) to
    enable per run stylingy.
    """

    def __init__(self, function):
        # enforce arugments
        self.finfo = FunctionInfo(function)
        fname = self.finfo.name
        # takes care of decorated functions
        self.decorated_function = self.finfo.function
        oargs = dict(zip(self.finfo.optional_arguments, self.finfo.defaults))
        errmsg = fname + ' has no optional argument "%s=None".'
        for a in ['output', 'ax']:
            assert a in oargs and oargs[a] is None, errmsg % a
        errmsg = fname + ' should start with "plot".'
        assert fname.startswith('plot') or fname == '__call__', errmsg
        assert self.finfo.kwargs, fname+' must accept **kwargs.'
        # attributes assigned during call
        callattr = ('project instance args kwargs ax figure savekwargs '
                    'output runs ax_parsed')
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

    def _interpret_args(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.runs = kwargs.pop('runs', None)
        self.output = kwargs.get('output')
        pax = kwargs.get('ax', None)
        self.ax_parsed = pax is not None
        self.ax = pax or plt.gca()
        self.kwargs['ax'] = self.ax
        self.figure = self.ax.get_figure()
        self._infer_project()

        self.savekwargs = {}
        self.savekwargs.update(self.project.save_figure_defaults)
        if type(self.output) is dict:
            op = self.output.pop('output', None)
            self.savekwargs.update(self.output)
            self.output = op
        return

    def __call__(self, *args, **kwargs):
        self._interpret_args(args, kwargs)

        if self.runs:
            result = self._plot_runs()
        else:
            result = self.decorated_function(*self.args, **self.kwargs)

        if self.output:
            save(self.output, self.figure, **self.savekwargs)
        # display if from commandline or browser api (if ax parsed it cant be
        # from the commandline or browser so must be a subcall)
        elif sys.argv[0].endswith('swimpy') and not self.ax_parsed:
            result = self._display_figure()
        return result

    def _plot_runs(self):
        ispi = self.instance.__class__ != self.project.__class__
        piname = self.instance.__class__.__name__  # project if not plugin
        # extract stings as labels for current
        current_label = None
        if hasattr(self.runs, '__iter__'):
            current_label = [r for r in self.runs if type(r) == str]
            if current_label:
                self.runs = [r for r in self.runs if r not in current_label]
                current_label = current_label[0]
        # transform runs to QuerySet
        runs = self.project.browser.runs.get_runs(self.runs)

        # plot current if current lable parsed
        if current_label:
            res = self.decorated_function(*self.args, label=current_label,
                                          **self.kwargs)
            result = [res]
        else:
            result = []

        for i, r in enumerate(runs):
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
            rkw['runs'] = (runs, i)
            rkw.setdefault('label', str(r))
            # call method with different instance as first argument as
            # decorated_function is unbound
            rre = pmeth.decorated_function(piinstance, *self.args[1:], **rkw)
            result.append(rre)
        # make sure a legend is shown if not already
        if self.ax.get_legend() is None:
            self.ax.legend()
        return result

    def _display_figure(self):
        # tight_layout doesnt work with irregular grid plots, lets try
        try:
            self.figure.tight_layout()
        except RuntimeError:
            warnings.warn('Figure doesnt allow tight layout.')
        # in Django API
        if len(sys.argv) > 1 and sys.argv[1] == 'browser':
            imgpath = tempfile.mkstemp()[1] + '.png'
            save(imgpath, self.figure, **self.savekwargs)
            self.figure.clear()
            return imgpath
        else:  # in CLI
            plt.show(block=True)
        return


def plot_many(functions, **kwall):
    """Plot mutiple plots in a grid.

    Arguments
    ---------
    functions : list (of lists) of callables or tuple(callable, dict)
        Plotting functions to call with either just defaults or with keyword
        arguments if a tuple with the callable and a dict is parsed.
        List in the list subdivides rows in the subplot grid. All functions
        must accept the ax keyword.
    kwall :
        Keywords applied to all functions.

    Returns
    -------
    axes
    """
    def norm_f(f):
        if callable(f):
            return (f, kwall)
        elif type(f) == tuple:
            assert len(f) == 2 and callable(f[0]) and type(f[1]) == dict
            f[1].update(kwall)
            return f
        elif type(f) == list:
            return [norm_f(i) for i in f]
        else:
            raise TypeError('functions entry %r is neither a ' % (f,) +
                            'callable or a tuple (callable, dict).')

    def call_f(f, ax, kw):
        try:
            f(ax=ax, **kw)
        except Exception:
            import traceback
            etype, ex, tb = sys.exc_info()
            raise etype('While calling %s, the below error occurred:\n\n' % f +
                        str(ex) + ':\n' + ''.join(traceback.format_tb(tb)))

    assert type(functions) == list
    nrow = len(functions)
    ncol = max([1]+[len(r) for r in functions if type(r) == list])
    grid = (nrow, ncol)
    axes = []
    normf = norm_f(functions)
    for irow, f in enumerate(normf):
        if type(f) == list:
            colspan = int(len(f)/float(ncol))
            for icol, (fu, kw) in enumerate(f):
                axes += [plt.subplot2grid(grid, (irow, icol), colspan=colspan)]
                call_f(fu, axes[-1], kw)
        else:
            axes += [plt.subplot2grid(grid, (irow, 0), colspan=ncol)]
            call_f(f[0], axes[-1], f[1])
    return axes


class plot_summary(object):
    """A plugin to enable project.plot_summary and run.plot."""

    plugin = ['__call__']

    def __init__(self, project, host=None):
        host = host or project
        self.host = host
        self.project = project
        return

    def _getattr(self, address):
        """Deep getattr with a dotted address."""
        a = self.host
        try:
            for i in address.split('.'):
                a = getattr(a, i)
        except AttributeError:
            print('Cant find %s, will be ignored.')
            return
        return a

    def _convert(self, l):
        """Recursively convert functions entries into callables or skip."""
        if type(l) == str:
            try:
                return self._getattr(l)
            except (AttributeError, IOError):
                return None
        elif type(l) == tuple:
            assert len(l) == 2 and type(l[0]) == str and type(l[1]) == dict
            e = self._convert(l[0])
            return (e, l[1]) if e else None
        elif type(l) == list:
            return [i for i in [self._convert(i) for i in l] if i]

    @parse_settings
    def __call__(self, functions=None, output=None, runs=None, ax=None, **kw):
        """Summary plot.

        Arguments
        ---------
        functions : list (of lists) of callables or tuple(callable, dict)
            Plotting functions to call with either just defaults or with
            keyword arguments if a tuple with the callable and a dict is
            parsed. List in the list subdivides rows in the subplot grid. All
            functions must accept the ax keyword. Mainly intended to be set in
            settings with the ``plot_summary_functions`` variable.
        kw :
            Keywords to all subplots.

        Returns
        -------
        list :
            Flat list of axes that were created.
        """
        plot_function = PlotFunction(self.__call__)
        pfkw = dict(ax=ax, output=output, runs=runs)
        plot_function._interpret_args([self], pfkw)

        normed_functions = self._convert(functions)
        if normed_functions:
            axes = plot_many(normed_functions, runs=runs, **kw)
        else:
            raise RuntimeError('No valid plots found in %s' % self.host)

        # remove axes legends and add figure legend
        legends = [l for l in [a.get_legend() for a in axes] if l is not None]
        if legends:
            plot_function.figure.legend(*axes[0].get_legend_handles_labels())
            [l.remove() for l in legends]

        # output/display
        if plot_function.output:
            save(plot_function.output, plot_function.figure,
                 **plot_function.savekwargs)
        # display if from commandline or browser api
        elif sys.argv[0].endswith('swimpy'):
            return plot_function._display_figure()
        return axes
