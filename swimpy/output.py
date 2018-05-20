"""
Result file interface.

All defined classes are attached to project and run instances as
propertyplugin that return a pandas.DataFrame. For files to be read from the
SWIM project, a from_project method needs to be defined. To read the data from
a run instance, a method refering to the extension of a file saved as
ResultFile needs to be defined (e.g. from_csv) or from_run to overwrite the
file selection.

Conventions:
------------
- class and method names should be lowercase, words separated by _ and
    descriptions should be singular (subbasin rather than subbasins)
- name word order: spatial domain (catchment, subbasin, hydrotope, station
    etc.), timestep adjective (daily, monthly, annually, average), variable
    and/or other descriptions. Pattern:
        domain_timestep_variable[_description...]
- all read from_* methods should parse **readkwargs to the pandas.read call
"""
import os.path as osp
import datetime as dt

import pandas as pd
from modelmanager.utils import propertyplugin

from swimpy import utils, plot

from matplotlib import pyplot as plt  # after plot

RESDIR = 'output/Res'
GISDIR = 'output/GIS'


@propertyplugin
class station_daily_discharge(utils.ProjectOrRunData):
    """
    Daily discharge of selected stations.
    """
    swim_path = osp.join(RESDIR, 'Q_gauges_sel_sub_routed_m3s.csv')
    plugin = ['plot']

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_csv(path, **readkwargs)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YEAR'), df.pop('DAY'))]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='d')
        return df

    @plot.plot_function
    def plot(self, ax=None, stations=None, regime=False,
             minmax=False, freq='d', output=None, **linekw):
        """Line plot of daily discharge of selected stations.

        Arguments
        ---------
        ax : <matplotlib.Axes>, optional
            Axes to plot to. Default is the current axes.
        stations: None | str | iterable
            Only show single (str) or subset (iterable) of stations. If None,
            show all found in file.
        regime : bool
            Plot regime. freq must be 'd' or 'm'.
        freq : <pandas frequency>
            Any pandas frequency to aggregate to.
        minmax : bool | dict
            Show min-max range if regime=True. Maybe a dictionary kwargs parsed
            to ax.fill_between.
        output : str path | dict
            Path to writeout or dict of keywords to parse to save_or_show.
        **linekw :
            Parse any keyword to the line plot function.
        """
        ax = ax or plt.gca()
        if stations is None:
            stations = self.columns[1:]  # first column is observed
        else:
            assert type(stations) == str or len(stations) > 0
            stations = [stations] if type(stations) == str else stations

        data = {}
        for st in ['mean'] + (['min', 'max'] if regime and minmax else []):
            data[st] = utils.aggregate_time(self[stations], regime=regime,
                                            freq=freq, regime_method=st)
        # show range first if required
        if regime and minmax:
            for s in stations:
                fbkw = minmax if type(minmax) == dict else {}
                fbkw.setdefault("alpha", 0.5)
                ax.fill_between(data['min'][s].index, data['max'][s], **fbkw)
        for s in stations:
            line = plot.plot_discharge(data['mean'][s], ax, **linekw)

        if regime:
            xlabs = {'d': 'Day of year', 'm': 'Month'}
            ax.set_xlabel(xlabs[freq])
        return line


@propertyplugin
class subbasin_daily_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(RESDIR, 'subd.prn')
    plugin = []

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_table(path, delim_whitespace=True, **readkwargs)
        dtms = [dt.date(1900 + y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YR'), df.pop('DAY'))]
        dtmspi = pd.PeriodIndex(dtms, freq='d', name='time')
        df.index = pd.MultiIndex.from_arrays([dtmspi, df.pop('SUB')])
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        pix = df.index.to_period(freq='d')
        df.index = pd.MultiIndex.from_arrays([pix, df.pop('SUB')])
        return df


@propertyplugin
class catchment_daily_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(RESDIR, 'bad.prn')

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_table(path, delim_whitespace=True, **readkwargs)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YR'), df.pop('DAY'))]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='d')
        return df


@propertyplugin
class catchment_monthly_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(RESDIR, 'bam.prn')

    @staticmethod
    def from_project(path, **readkwargs):
        with open(path, 'r') as f:
            iyr = int(f.readline().strip().split('=')[1])
            df = pd.read_table(f, delim_whitespace=True, index_col=False,
                               **readkwargs)
        df.dropna(inplace=True)  # exclude Year = ...
        df = df.drop(df.index[range(12, len(df), 12+1)])  # excluded headers
        dtms = ['%04i-%02i' % (iyr+int((i-1)/12.), m)
                for i, m in enumerate(df.pop('MON').astype(int))]
        df.index = pd.PeriodIndex(dtms, freq='m', name='time')
        return df.astype(float)

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='m')
        return df


@propertyplugin
class catchment_annual_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(RESDIR, 'bay.prn')
    plugin = ['plot_mean', 'print_mean']

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_table(path, delim_whitespace=True, index_col=0,
                           parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='a')
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='a')
        return df

    @plot.plot_function
    def plot_mean(self, ax=None, output=None):
        bars = plot.plot_waterbalance(self.mean(), ax=ax)
        return bars

    def print_mean(self):
        mean = self.mean().to_string()
        print(mean)
        return mean


# only import the property plugins on from output import *
__all__ = [n for n, p in globals().items() if isinstance(p, propertyplugin)]
