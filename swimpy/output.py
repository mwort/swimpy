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
import sys

import datetime as dt

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from swimpy import utils, plot


RESDIR = 'output/Res'
GISDIR = 'output/GIS'


class station_daily_discharge(utils.ProjectOrRunData):
    """
    Daily discharge of selected stations.
    """
    swim_path = osp.join(RESDIR, 'Q_gauges_sel_sub_routed_m3s.csv')
    plugin_functions = []

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


class subbasin_daily_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(RESDIR, 'subd.prn')
    plugin_functions = []

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


class catchment_annual_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(RESDIR, 'bay.prn')
    plugin_functions = ['plot_mean', 'print_mean']

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

    def plot_mean(self, ax=plt.gca(), output=None, size=(160, 125)):
        bars = plot.plot_mean_waterbalance(self)
        plot.save_or_show(output=output, **(dict(size=size) if size else {}))
        return bars

    def print_mean(self):
        mean = self.mean()
        print(mean.to_string())
        return
