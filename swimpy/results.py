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
- names should be lowercase, words separated by _, singular descriptions
- name word order: spatial domain (catchment, subbasin, hydrotope, station
    etc.), timestep adjective (daily, monthly, annually, average), variable
    and/or other descriptions. Pattern:
        domain_timestep_variable[_description...]
"""
import os.path as osp
import datetime as dt

import pandas as pd

from modelmanager import utils as mmutils
from swimpy import utils


resdir = 'output/Res'
gisdir = 'output/GIS'


@mmutils.propertyplugin
class station_daily_discharge_routed(utils.ProjectOrRunData):
    swim_path = osp.join(resdir, 'Q_gauges_sel_sub_routed_m3s.csv')
    plugin_functions = []

    def from_project(self):
        df = pd.read_csv(self.path)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YEAR'), df.pop('DAY'))]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        return df

    def from_csv(self):
        df = pd.read_csv(self.path, index_col=0, parse_dates=[0])
        df.index = df.index.to_period(freq='d')
        return df


@mmutils.propertyplugin
class subbasin_daily_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(resdir, 'subd.prn')
    plugin_functions = []

    def from_project(self):
        df = pd.read_table(self.path, delim_whitespace=True)
        dtms = [dt.date(1900 + y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YR'), df.pop('DAY'))]
        dtmspi = pd.PeriodIndex(dtms, freq='d', name='time')
        df.index = pd.MultiIndex.from_arrays([dtmspi, df.pop('SUB')])

        return df

    def from_csv(self):
        df = pd.read_csv(self.path, index_col=0, parse_dates=[0])
        pix = df.index.to_period(freq='d')
        df.index = pd.MultiIndex.from_arrays([pix, df.pop('SUB')])
        return df


@mmutils.propertyplugin
class catchment_daily_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(resdir, 'bad.prn')

    def from_project(self):
        df = pd.read_table(self.path, delim_whitespace=True)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YR'), df.pop('DAY'))]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        return df

    def from_csv(self):
        df = pd.read_csv(self.path, index_col=0, parse_dates=[0])
        df.index = df.index.to_period(freq='d')
        return df


@mmutils.propertyplugin
class catchment_monthly_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(resdir, 'bam.prn')

    def from_project(self):
        with open(self.path, 'r') as f:
            iyr = int(f.readline().strip().split('=')[1])
            df = pd.read_table(f, delim_whitespace=True, index_col=False)
        df.dropna(inplace=True)  # exclude Year = ...
        df = df.drop(df.index[range(12, len(df), 12+1)])  # excluded headers
        dtms = ['%04i-%02i' % (iyr+int((i-1)/12.), m)
                for i, m in enumerate(df.pop('MON').astype(int))]
        df.index = pd.PeriodIndex(dtms, freq='m', name='time')
        return df.astype(float)

    def from_csv(self):
        df = pd.read_csv(self.path, index_col=0, parse_dates=[0])
        df.index = df.index.to_period(freq='m')
        return df


@mmutils.propertyplugin
class catchment_annual_waterbalance(utils.ProjectOrRunData):
    swim_path = osp.join(resdir, 'bay.prn')

    def from_project(self):
        df = pd.read_table(self.path, delim_whitespace=True,
                           index_col=0, parse_dates=[0])
        df.index = df.index.to_period(freq='a')
        return df

    def from_csv(self):
        df = pd.read_csv(self.path, index_col=0, parse_dates=[0])
        df.index = df.index.to_period(freq='a')
        return df
