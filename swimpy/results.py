"""
Result file interface.

All defined classes are attached to project and run instances as
propertyplugin that return a pandas.DataFrame. For files to be read from the
SWIM project, a from_project method needs to be defined. To read the data from
a run instance, a method refering to the extension of a file saved as
ResultFile needs to be defined (e.g. from_csv) or from_run to overwrite the
file selection.
"""
import datetime as dt
import inspect

import pandas as pd

import modelmanager
from swimpy import utils


class routed_station_discharge(utils.ProjectOrRunData):
    swim_path = 'output/Res/Q_gauges_sel_sub_routed_m3s.csv'
    plugin_functions = []

    def from_project(self, path=None):
        df = pd.read_csv(path or self.path)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YEAR'), df.pop('DAY'))]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        return df

    def from_csv(self, path=None):
        df = pd.read_csv(path or self.path, index_col=0, parse_dates=[0])
        df.index = df.index.to_period(freq='d')
        return df


class daily_subbasin_waterbalance(utils.ProjectOrRunData):
    swim_path = 'output/Res/subd.prn'
    plugin_functions = []

    def from_project(self, path=None):
        df = pd.read_table(path or self.path, delim_whitespace=True)
        dtms = [dt.date(1900 + y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YR'), df.pop('DAY'))]
        dtmspi = pd.PeriodIndex(dtms, freq='d', name='time')
        df.index = pd.MultiIndex.from_arrays([dtmspi, df.pop('SUB')])

        return df

    def from_csv(self, path=None):
        df = pd.read_csv(path or self.path, index_col=0, parse_dates=[0])
        pix = df.index.to_period(freq='d')
        df.index = pd.MultiIndex.from_arrays([pix, df.pop('SUB')])
        return df


properties = {n: modelmanager.utils.propertyplugin(obj)
              for n, obj in locals().items()
              if (not n.startswith('_') and inspect.isclass(obj)
                  and utils.ProjectOrRunData in obj.__bases__)}
