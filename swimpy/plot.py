"""
SWIM related plotting functions.

Standalone functions to create plots for SWIM input/output. They are used
throught the SWIMpy package but collected here to enable reuse.

All functions should accept an optional ax=None argument to plot to, i.e.
defaulting to the current axes (ax = ax or plt.gca()).
"""
import sys
import tempfile
from functools import wraps

from modelmanager.settings import FunctionInfo
import matplotlib as mpl
if len(sys.argv) > 1 and sys.argv[1] == 'browser':
    mpl.use('Agg')
import matplotlib.pyplot as plt


SAVEFIG_DEFAULTS = dict(
    bbox_inches='tight',
    pad_inches=0.03,
    orientation='portrait',
    dpi=200,
    size=(180, 120),  # mm
)


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
    for d, v in SAVEFIG_DEFAULTS.items():
        savekwargs.setdefault(d, v)
    size = savekwargs.pop('size')
    assert len(size) == 2, 'size must be (width, height) not %r' % size
    mmpi = 25.4
    figure.set_size_inches(size[0]/mmpi, size[1]/mmpi)  # (width, hight)
    figure.savefig(output, **savekwargs)
    return


def plot_function(function):
    """A decorator to enforce and handle generic plot function tasks.

    - enforces name starting with 'plot'.
    - enforces output=None and ax=None arugments.
    - allows saving figure to file with output argument that may either be
      string path or a dict with kwargs to save.
    - displays interactive plot if executed from commandline.
    - saves current figure to a temp path when executed in browser API.
    """
    finfo = FunctionInfo(function)
    oargs = dict(zip(finfo.optional_arguments, finfo.defaults))
    errmsg = finfo.name + ' has no optional argument "%s=None".'
    assert 'output' in oargs and oargs['output'] is None, errmsg % 'output'
    assert 'ax' in oargs and oargs['ax'] is None, errmsg % 'ax'
    errmsg = finfo.name + ' should start with "plot".'
    assert finfo.name.startswith('plot'), errmsg

    @wraps(function)
    def f(*args, **kwargs):
        result = function(*args, **kwargs)
        # unpack savekwargs
        savekwargs = {}
        output = kwargs.get('output', None)
        ax = kwargs.get('ax', None)
        figure = ax.get_figure() if ax else plt.gcf()
        if type(output) is dict:
            op = output.pop('output', None)
            savekwargs.update(output)
            output = op
        # save to file
        if output:
            save(output, figure, **savekwargs)
        # display if from commandline or browser api
        elif sys.argv[0].endswith('swimpy'):
            # in Django API
            if len(sys.argv) > 1 and sys.argv[1] == 'browser':
                imgpath = tempfile.mkstemp()[1] + '.png'
                save(imgpath, figure, **savekwargs)
                figure.clear()
                return imgpath
            # in CLI
            plt.show(block=True)
        return result

    f.decorated_function = function
    # add signiture if PY2
    if sys.version_info < (3, 0):
        f.__doc__ = '%s(%s)\n' % (finfo.name, finfo.signiture) + finfo.doc
    return f


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


def _index_to_timestamp(index):
    """Convert a pandas index to timestamps if needed.
    Needed to parse pandas PeriodIndex to pyplot plotting functions."""
    return index.to_timestamp() if hasattr(index, 'to_timestamp') else index
