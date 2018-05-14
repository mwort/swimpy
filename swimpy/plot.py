"""
SWIM related plotting functions.

Standalone functions to create plots for SWIM input/output. They are used
throught the SWIMpy package but collected here to enable reuse.

All functions should accept an optional ax=plt.gca() argument to plot to, i.e.
defaulting to the current axes.
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

SAVEFIG_DEFAULTS = dict(
    bbox_inches='tight',
    pad_inches=0.03,
    orientation='portrait',
    dpi=300,
    size=(160, 125),  # mm
)


def save(output, figure=plt.gcf(), **savekwargs):
    """Convenience function to set figure size and save a matplotlib figure.

    Arguements:
    -----------
    output : str | dict
        Path to save figure to. Extension determines format. May also be a dict
        of savekwargs including an ``output`` keyword with an output path.
        See **savekwargs for options.
    figure : matplotlib.Figure object, optional
        Defaults to current figure.
    **savekwargs :
        Any keyword argument parsed to figure.savefig() method. Special keys:
        ``size`` : len 2 tuple
            Size in mm.

    Returns: None
    """
    if type(output) is dict:
        op = output.pop('output')
        savekwargs.update(output)
        output = op
    assert type(output) == str, 'output %r must be string path.'
    for d, v in SAVEFIG_DEFAULTS.items():
        savekwargs.setdefault(d, v)
    size = savekwargs.pop('size')
    assert len(size) == 2, 'size must be (width, height) not %r' % size
    mmpi = 25.4
    figure.set_size_inches(size[0]/mmpi, size[1]/mmpi)  # (width, hight)
    figure.savefig(output, **savekwargs)
    return


def save_or_show(output=None, figure=plt.gcf(), **savekwargs):
    """Convenience function to set figure size and save (if output given) or
    show figure (if run from commandline).

    Arguements:
    -----------
    output : None | str | dict, optional
        Path to save figure to. Extension determines format. May also be a dict
        of savekwargs including an ``output`` keyword with an output path.
        swimpy.plot.save options.
    figure : matplotlib.Figure object, optional
        Defaults to current figure.
    **savekwargs :
        Any keyword argument parsed to swimpy.plot.save()

    Returns: None
    """
    if output:
        save(output, figure, **savekwargs)
    # if from commandline, show figure
    elif sys.argv[0].endswith('swimpy'):
        plt.show(block=True)
    return


def plot_waterbalance(series, ax=plt.gca(), **barkwargs):
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
    bars = series.plot.bar(ax=ax, **barkwargs)
    ax.set_ylabel('mm per year')
    ax.set_title('Catchment mean water balance')
    return bars


def plot_temperature_range(series, ax=plt.gca(), minmax=[], **linekwargs):
    """Plot temperature with optional min-max range."""
    assert len(minmax) in [0, 2]
    if minmax:
        kw = dict(alpha=0.3, color='k')
        mmfill = ax.fill_between(_index_to_timestamp(series.index), minmax[0],
                                 minmax[1], **kw)
    line = ax.plot(_index_to_timestamp(series.index), series, **linekwargs)
    ax.set_ylabel('Temperature [C]')
    ax.set_xlabel('Time')
    return (line, mmfill) if minmax else line


def plot_precipitation_bars(series, ax=plt.gca(), **barkwargs):
    """Plot precipitation as bars."""
    if hasattr(series.index, 'to_timestamp'):
        freqstr = series.index.freqstr.split('-')[0][-1].lower()  # last letter
        width = {'a': 365, 'm': series.index.days_in_month, 'd': 1}
        barkwargs.setdefault('width', width[freqstr]*0.8)
    bars = ax.bar(_index_to_timestamp(series.index), series, **barkwargs)
    ax.set_ylabel('Precipitation [mm]')
    ax.set_xlabel('Time')
    return bars


def _index_to_timestamp(index):
    """Convert a pandas index to timestamps if needed.
    Needed to parse pandas PeriodIndex to pyplot plotting functions."""
    return index.to_timestamp() if hasattr(index, 'to_timestamp') else index
