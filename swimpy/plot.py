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
)


def save_or_show(output=None, figure=plt.gcf(), size=(160, 125), **savekwargs):
    """Convenience function to set figure size and save (if output given) or
    show figure (if run from commandline).

    Arguements:
    -----------
    output : None | str, optional
        Path to save figure to. Extension determines format.
    size : len 2 tuple, optional
        Size in mm.
    figure : matplotlib.Figure object, optional
        Defaults to current figure.
    **savekwargs :
        Any keyword argument parsed to figure.savefig() method.

    Returns: None
    """
    if output:
        assert type(output) == str, 'output %r must be string path.'
        assert len(size) == 2, 'size must be (width, height) not %r' % size
        mmpi = 25.4
        figure.set_size_inches(size[0]/mmpi, size[1]/mmpi)  # (width, hight)
        for d, v in SAVEFIG_DEFAULTS.items():
            savekwargs.setdefault(d, v)
        figure.savefig(output, **savekwargs)
        figure.clear
    elif sys.argv[0].endswith('swimpy'):  # if from commandline, show figure
        plt.show(block=True)
    return


def plot_mean_waterbalance(df, ax=plt.gca(), **barkwargs):
    """Bar plot of water balance terms.

    Arguments:
    ----------
    df : pd.DataFrame
        Will be averaged over index, column names will be used as x labels.
    ax : plt.Axes, optional
        An axes to plot to. If None given, the current axes are used.
    **barkwargs :
        plt.bar keyword arguments.

    Returns: bars
    """
    means = df.mean()
    bars = means.plot.bar(ax=ax, **barkwargs)
    ax.set_ylabel('mm per year')
    ax.set_title('Catchment mean water balance')
    return bars
