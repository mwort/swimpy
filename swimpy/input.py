"""
SWIM input functionality.
"""
import os.path as osp

import pandas as pd
from modelmanager.utils import propertyplugin as _propertyplugin
from modelmanager.plugins.templates import TemplatesDict as _TemplatesDict

from swimpy import utils, plot


@_propertyplugin
class basin_parameters(_TemplatesDict):
    """
    Set or get any values from the .bsn file by variable name.
    """
    template_patterns = ['input/*.bsn']


@_propertyplugin
class config_parameters(_TemplatesDict):
    """
    Set or get any values from the .cod or swim.conf file by variable name.
    """
    template_patterns = ['input/*.cod', 'swim.conf']


@_propertyplugin
class subcatch_parameters(utils.ReadWriteDataFrame):
    """
    Read or write parameters in the subcatch.prm file.
    """
    path = 'input/subcatch.prm'

    def read(self, **kwargs):
        bsn = pd.read_table(self.path, delim_whitespace=True)
        stn = 'stationID' if 'stationID' in bsn.columns else 'station'
        bsn.set_index(stn, inplace=True)
        pd.DataFrame.__init__(self, bsn)
        return

    def write(self, **kwargs):
        bsn = self.copy()
        bsn['stationID'] = bsn.index
        strtbl = bsn.to_string(index=False, index_names=False)
        with open(self.path, 'w') as f:
            f.write(strtbl)
        return


class climate(object):
    """All climate input related functionality."""

    def __init__(self, project):
        self.project = project
        return

    @_propertyplugin
    class inputdata(utils.ReadWriteDataFrame):
        """A lazy DataFrame representation of the two 'clim'-files.

        Rather than being read on instantiation, .read() and .write() need to
        be called explicitly since both operations are commonly time-consuming.
        """
        namepattern = 'clim%i.dat'
        variables = ['radiation', 'humidity', 'precipitation',
                     'tmin', 'tmax', 'tmean']
        clim_variables = {1: variables[:3], 2: variables[3:]}
        column_levels = ['variable', 'subbasinID']
        plugin = ['print_stats', 'plot_temperature', 'plot_precipitation']

        def __init__(self, project):
            pd.DataFrame.__init__(self)
            self.project = project
            self.path = project.config_parameters['climatedir']
            utils.ReadWriteDataFrame.__init__(self, project)
            return

        def read(self, climdir=None, **kw):
            startyr = self.project.config_parameters['iyr']
            path = osp.join(climdir or self.path, self.namepattern)
            dfs = pd.concat([self.read_clim(path % i, startyr, vs, **kw)
                             for i, vs in self.clim_variables.items()], axis=1)
            dfs.sort_index(axis=1, inplace=True)
            pd.DataFrame.__init__(self, dfs)
            return self

        @classmethod
        def read_clim(cls, path, startyear, variables, **readkwargs):
            """Read single clim file and return DataFrame with index and
            columns.
            """
            assert len(variables) == 3
            readargs = dict(delim_whitespace=True, header=None, skiprows=1)
            readargs.update(readkwargs)
            df = pd.read_table(path, **readargs)
            df.index = pd.PeriodIndex(start=str(startyear), periods=len(df),
                                      freq='d', name='time')
            nsub = int(len(df.columns)/3)
            df.columns = cls._create_columns(nsub, variables)
            return df

        @classmethod
        def _create_columns(cls, nsubbasins, variables):
            v = [range(1, nsubbasins+1), variables]
            ix = pd.MultiIndex.from_product(v, names=cls.column_levels[::-1])
            return ix.swaplevel()

        def write(self, outdir=None, **writekw):
            path = osp.join(outdir or self.path, self.namepattern)
            for i, vs in self.clim_variables.items():
                # enforce initial column order
                df = self[self._create_columns(int(len(self.columns)/6), vs)]
                header = ['%s_%s' % (v[:4], s) for v, s in df.columns]
                writeargs = dict(index=False, header=header)
                writeargs.update(writekw)
                with open(path % i, 'w') as f:
                    df.to_string(f, **writeargs)
            return

        def print_stats(self):
            """Print statistics for all variables."""
            stats = self.mean(axis=1, level=0).describe().round(2).to_string()
            print(stats)
            return stats

        def aggregate(self, variables=[], **kw):
            """Mean data over all subbasins and optionally subset and aggregate
            to a frequency or regime.

            Arguments
            ---------
            variables : list
                Subset variables. If empty or None, return all.
            **kw :
                Keywords to utils.aggregate_time.
            """
            vars = variables or self.variables
            subs = self[vars].mean(axis=1, level='variable')
            aggm = {v: 'sum' if v == 'precipitation' else 'mean' for v in vars}
            aggregated = utils.aggregate_time(subs, resample_method=aggm, **kw)
            return aggregated

        @plot.plot_function
        def plot_temperature(self, ax=None, regime=False, freq='d',
                             minmax=True, output=None, **linekw):
            """Line plot of mean catchment temperature.

            Arguments
            ---------
            ax : <matplotlib.Axes>, optional
                Axes to plot to. Default is the current axes.
            regime : bool
                Plot regime. freq must be 'd' or 'm'.
            freq : <pandas frequency>
                Any pandas frequency to aggregate to.
            minmax : bool
                Show min-max range.
            output : str path | dict
                Path to writeout or dict of keywords to parse to save_or_show.
            **kw :
                Parse any keyword to the tmean line plot function.
            """
            clim = self.aggregate(variables=['tmean', 'tmin', 'tmax'],
                                  freq=freq, regime=regime)
            minmax = [clim.tmin, clim.tmax] if minmax else []
            line = plot.plot_temperature_range(clim.tmean, ax, minmax=minmax,
                                               **linekw)
            if regime:
                xlabs = {'d': 'Day of year', 'm': 'Month'}
                ax.set_xlabel(xlabs[freq])
            return line

        @plot.plot_function
        def plot_precipitation(self, ax=None, regime=False, freq='d',
                               output=None, **barkwargs):
            """Bar plot of mean catchment precipitation.

            Arguments
            ---------
            ax : <matplotlib.Axes>, optional
                Axes to plot to. Default is the current axes.
            regime : bool
                Plot regime. freq must be 'd' or 'm'.
            freq : <pandas frequency>
                Any pandas frequency to aggregate to.
            output : str path | dict
                Path to writeout or dict of keywords to parse to save_or_show.
            **barkwargs :
                Parse any keyword to the bar plot function.
            """
            clim = self.aggregate(variables=['precipitation'],
                                  freq=freq, regime=regime)['precipitation']
            bars = plot.plot_precipitation_bars(clim, ax, **barkwargs)
            if regime:
                xlabs = {'d': 'Day of year', 'm': 'Month'}
                ax.set_xlabel(xlabs[freq])
            return bars
