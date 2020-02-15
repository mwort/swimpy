"""
Output/model results file interfaces.

All defined classes are attached to project and run instances as
propertyplugin that return a pandas.DataFrame. For files to be read from the
SWIM project, a from_project method needs to be defined. To read the data from
a run instance, a method refering to the extension of a file saved as
run file needs to be defined (e.g. from_csv) or from_run to overwrite the
file selection.

Conventions
-----------
- class and method names should be lowercase, words separated by _ and
  descriptions should be singular (subbasin rather than subbasins)
- name word order: spatial domain (catchment, subbasin, hydrotope, station
  etc.), timestep adjective (daily, monthly, annually, average), variable
  and/or other descriptions. Pattern: domain_timestep_variable[_description...]
- all read ``from_*`` methods should parse any keyword to the pandas.read call
"""
import os.path as osp
from glob import glob
import datetime as dt
import calendar
import warnings
import inspect

import numpy as np
import pandas as pd
from modelmanager.utils import propertyplugin
from modelmanager.plugins.pandas import ProjectOrRunData

from swimpy import utils, plot, hydro
from swimpy.plot import plot_function as _plot_function
from swimpy.grass import _subbasin_or_hydrotope_values_to_raster


RESDIR = 'output/Res'
GISDIR = 'output/GIS'


class station_daily_discharge(ProjectOrRunData):
    """
    Daily discharge of selected stations.
    """
    path = osp.join(RESDIR, 'Q_gauges_sel_sub_routed_m3s.csv')
    plugin = ['plot', 'plot_regime', 'plot_flow_duration_polar']

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

    def _default_stations(self, stations=None):
        if stations is None:
            dstations = self.columns[1:]  # first column is observed
        else:
            assert type(stations) == str or len(stations) > 0
            dstations = [stations] if type(stations) == str else stations
        return dstations

    @_plot_function
    def plot(self, stations=None, freq='d', minmax=False,
             observed=False, ax=None, runs=None, output=None, **linekw):
        """Line plot of daily discharge of selected stations.

        Arguments
        ---------
        stations : None | str | iterable
            Only show single (str) or subset (iterable) of stations. If None,
            show all found in file.
        freq : <pandas frequency>
            Any pandas frequency to aggregate to.
        observed : bool
            Add line for observed discharge. stations.daily_discharge_observed
            must be configured.
        **linekw :
            Parse any keyword to the line plot function.
        """
        stations = self._default_stations(stations)
        data = utils.aggregate_time(self[stations], freq=freq)
        # plot observed
        if observed:
            obs = utils.aggregate_time(
                (self.project.stations.daily_discharge_observed
                 .loc[self.index, stations]), freq=freq)
            clrs = plot.default_colors(len(stations), linekw.get('colors', []))
            for c, s in zip(clrs, stations):
                plot.plot_discharge(obs[s], ax, linestyle='--', color=c)
        for s in stations:
            # add label if multiple runs
            if runs and len(runs[0]) > 1:
                qs, i = runs
                lab = '%s ' % qs[i] + ('' if len(stations) == 1 else str(s))
                linekw['label'] = lab
            line = plot.plot_discharge(data[s], ax, **linekw)
        return line

    @_plot_function
    def plot_regime(self, stations=None, freq='d', minmax=False,
                    observed=False, ax=None, runs=None, output=None, **linekw):
        """Line plot of daily discharge of selected stations.

        Arguments
        ---------
        stations : None | str | iterable
            Only show single (str) or subset (iterable) of stations. If None,
            show all found in file.
        freq : str
            Regime frequency, d (daily) or m (monthly).
        minmax : bool | dict
            Show min-max range. May be a dictionary kwargs
            parsed to ax.fill_between.
        observed : bool
            Add line for observed discharge. stations.daily_discharge_observed
            must be configured.
        **linekw :
            Parse any keyword to the line plot function.
        """
        stations = self._default_stations(stations)
        data = {}
        for st in ['mean'] + (['min', 'max'] if minmax else []):
            data[st] = utils.aggregate_time(self[stations], regime=True,
                                            freq=freq, regime_method=st)
        # show range first if required
        if minmax:
            for s in stations:
                fbkw = minmax if type(minmax) == dict else {}
                fbkw.setdefault("alpha", 0.5)
                ax.fill_between(data['min'][s].index, data['max'][s], **fbkw)
        # plot observed
        if observed:
            obs = utils.aggregate_time(
                (self.project.stations.daily_discharge_observed
                 .loc[self.index, stations]), regime=True, freq=freq)
            clrs = plot.default_colors(len(stations), linekw.get('colors', []))
            for c, s in zip(clrs, stations):
                plot.plot_discharge(obs[s], ax, linestyle='--', color=c)
        for s in stations:
            # add label if multiple runs
            if runs and len(runs[0]) > 1:
                qs, i = runs
                lab = '%s ' % qs[i] + ('' if len(stations) == 1 else str(s))
                linekw['label'] = lab
            line = plot.plot_discharge(data['mean'][s], ax, **linekw)

        xlabs = {'d': 'Day of year', 'm': 'Month'}
        ax.set_xlabel(xlabs[freq])
        if freq == 'm':
            ax.set_xticklabels([s[0] for s in calendar.month_abbr[1:]])
            ax.set_xticks(range(1, 12+1))
            ax.set_xlim(1, 12)
        elif freq == 'd':
            nd = np.array(calendar.mdays).cumsum()
            nd[:-1] += 1
            ax.set_xticks(nd)
            ax.set_xlim(1, 365)
        return line

    @_plot_function
    def plot_flow_duration(self, stations=None, ax=None, runs=None,
                           output=None, **linekw):
        stations = self._default_stations(stations)
        lines = []
        for s in stations:
            fd = hydro.flow_duration(self[s])
            line = plot.plot_flow_duration(fd, ax=ax, **linekw)
            lines.append(line)
        return

    @_plot_function
    def plot_flow_duration_polar(self, station, percentilestep=10, freq='m',
                                 colormap='jet_r', ax=None, runs=None,
                                 output=None, **barkw):
        """Plot flow duration on a wheel of month or days of year.

        Arguments
        ---------
        station : str
            A single station label (not possible for multiple stations).
        percentilestep : % <= 50
            Intervals of flow duration of 100%.
        freq : 'm' | 'd'
            Duration per month or day of year.
        colormap : str
            Matplotlib to use for the colour shading.
        """
        if runs:
            assert len(runs[0]) == 1
        ax = plot.plot_flow_duration_polar(self[station], freq=freq, ax=ax,
                                           percentilestep=percentilestep,
                                           colormap=colormap, **barkw)
        return ax

    def peak_over_threshold(self, percentile=1, threshold=None, maxgap=None,
                            stations=None):
        """Identify peaks over a threshold, return max, length, date and recurrence.

        Arguments
        ---------
        percentile : number
            The percentile threshold of q., e.g. 1 means Q1.
        threshold : number, optional
            Absolute threshold to use for peak identification.
        maxgap : int, optional
            Largest gap between two threshold exceedance periods to count as
            single flood event. Number of timesteps. If not given, every
            exceedance is counted as individual flood event.
        stations : stationID | list of stationIDs
            Return subset of stations. Default all.

        Returns
        -------
        pd.DataFrame :
            Peak discharge ordered dataframe with order index and peak q,
            length, peak date and recurrence columns with MultiIndex if more
            than one station is selected.
        """
        stations = self._default_stations(stations)
        kw = dict(percentile=percentile, threshold=threshold, maxgap=maxgap)
        pot = [hydro.peak_over_threshold(self[s], **kw) for s in stations]
        return pot[0] if len(stations) == 1 else pd.concat(pot, keys=stations)

    def obs_sim_overlap(self, warmupyears=1):
        """Return overlapping obs and sim dataframes excluding warmup period.

        Arguments
        ---------
        warmupyears : int
            Number of years to skip at beginng as warm up period.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame) : observed and simulated discharge.
        """
        obs = self.project.stations.daily_discharge_observed
        # exclude warmup period
        sim = self[str(self.index[0].year+warmupyears):]
        obsa, sima = obs.align(sim, join='inner')
        return obsa, sima

    @property
    def NSE(self):
        """pandas.Series of Nash-Sutcliff efficiency excluding warmup year."""
        obs, sim = self.obs_sim_overlap()
        return pd.Series({s: hydro.NSE(obs[s], sim[s]) for s in obs.columns})

    @property
    def rNSE(self):
        """pandas.Series of reverse Nash-Sutcliff efficiency (best = 0)"""
        return 1 - self.NSE

    @property
    def pbias(self):
        """pandas.Series of percent bias excluding warmup year."""
        obs, sim = self.obs_sim_overlap()
        return pd.Series({s: hydro.pbias(obs[s], sim[s]) for s in obs.columns})

    @property
    def pbias_abs(self):
        """pandas.Series of absolute percent bias excluding warmup year."""
        return self.pbias.abs()


class subbasin_daily_waterbalance(ProjectOrRunData):
    path = osp.join(RESDIR, 'subd.prn')
    plugin = ['to_raster']

    @staticmethod
    def from_project(path, **readkwargs):
        def parse_time(y, d):
            dto = dt.date(int(y), 1, 1) + dt.timedelta(int(d) - 1)
            return pd.Period(dto, freq='d')
        d = pd.read_csv(path, delim_whitespace=True, date_parser=parse_time,
                        parse_dates=[[0, 1]], index_col=[0, 1], **readkwargs)
        d.index.names = ['time', 'subbasinID']
        return d

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=[0, 1], parse_dates=[0],
                         date_parser=pd.Period, **readkwargs)
        return df

    def to_raster(self, variable, timestep=None, prefix=None, name=None,
                  strds=True, mapset=None):
        # extra argument
        """variable : str
            Selected variable (will be appended to default prefix).
        """
        prefix = prefix or self.__class__.__name__ + '_' + variable.lower()
        _subbasin_or_hydrotope_values_to_raster(
            self.project, self[variable].unstack(),
            self.project.subbasins.reclass, timestep=timestep, name=name,
            prefix=prefix, strds=strds, mapset=mapset)
        return
    to_raster.__doc__ = (_subbasin_or_hydrotope_values_to_raster.__doc__ +
                         to_raster.__doc__)


class subbasin_monthly_waterbalance(subbasin_daily_waterbalance):
    path = osp.join(RESDIR, 'subm.prn')

    def from_project(self, path, **readkwargs):
        styr = self.project.config_parameters['iyr']

        def parse_time(y, m):
            return pd.Period('%04i-%02i' % (styr+int(y)-1, int(m)), freq='m')
        with open(path) as f:
            header = f.readline().split()
            df = pd.read_csv(f, delim_whitespace=True, skiprows=1, header=None,
                             index_col=[0, 1], date_parser=parse_time,
                             parse_dates=[[0, 1]], names=header, **readkwargs)
        df.index.names = ['time', 'subbasinID']
        return df


class subbasin_daily_discharge(ProjectOrRunData):
    path = osp.join(RESDIR, 'Q_gauges_all_sub_routed_m3s.csv')

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_csv(path, delim_whitespace=True, index_col=[0, 1],
                         **readkwargs)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1) for y, d in df.index]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        df.columns = df.columns.astype(int)
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='d')
        df.columns = df.columns.astype(int)
        return df


class subbasin_daily_runoff(subbasin_daily_discharge):
    path = osp.join(RESDIR, 'Q_gauges_all_sub_mm.csv')


class catchment_daily_waterbalance(ProjectOrRunData):
    path = osp.join(RESDIR, 'bad.prn')

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_csv(path, delim_whitespace=True, **readkwargs)
        dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YR'), df.pop('DAY'))]
        df.index = pd.PeriodIndex(dtms, freq='d', name='time')
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='d')
        return df


class catchment_monthly_waterbalance(ProjectOrRunData):
    path = osp.join(RESDIR, 'bam.prn')

    @staticmethod
    def from_project(path, **readkwargs):
        with open(path, 'r') as f:
            iyr = int(f.readline().strip().split('=')[1])
            df = pd.read_csv(f, delim_whitespace=True, index_col=False,
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


class catchment_annual_waterbalance(ProjectOrRunData):
    path = osp.join(RESDIR, 'bay.prn')
    plugin = ['plot_mean', 'print_mean']

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_csv(path, delim_whitespace=True, index_col=0,
                         parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='a')
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='a')
        return df

    @plot.plot_function
    def plot_mean(self, ax=None, runs=None, output=None, **barkw):
        bars = plot.plot_waterbalance(self.mean(), ax=ax, **barkw)
        return bars

    def print_mean(self):
        mean = self.mean().to_string()
        print(mean)
        return

    @property
    def runoff_coefficient(self):
        (_, p), (_, r) = self[['PREC', '3Q']].items()
        return r/p


class subcatch_annual_waterbalance(ProjectOrRunData):
    path = osp.join(RESDIR, 'bay_sc.csv')
    plugin = ['print_mean']

    @staticmethod
    def from_project(path, **readkwargs):
        df = pd.read_csv(path, index_col=[0, 1], parse_dates=[1], **readkwargs)
        api = df.index.levels[1].to_period(freq='a')
        df.index.set_levels(api, level=1, inplace=True)
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=[0, 1], parse_dates=[1], **readkwargs)
        api = df.index.levels[1].to_period(freq='a')
        df.index.set_levels(api, level=1, inplace=True)
        return df

    @property
    def runoff_coefficient(self):
        (_, p), (_, r) = self[['PREC', '3Q']].items()
        return (r/p).unstack(level=0)

    def print_mean(self, catchments=None):
        """Print average values. Selected catchments or all (default)."""
        df = self.loc[catchments] if catchments else self
        ml = 0 if hasattr(df.index, 'levels') else None
        mdf = df.mean(level=ml).T
        print(mdf.to_string())
        return mdf


class hydrotope_daily_waterbalance(ProjectOrRunData):
    path = osp.join(RESDIR, 'htp.prn')

    @staticmethod
    def from_project(path, **readkwargs):
        args = dict(
            delim_whitespace=True, index_col=[0, 1, 2], parse_dates=[[0, 1]],
            date_parser=lambda y, d: dt.datetime.strptime(y+'-'+d, '%Y-%j'))
        args.update(readkwargs)
        htp = pd.read_csv(path, **args)
        htp.index.set_levels(htp.index.levels[0].to_period(), 0, inplace=True)
        htp.index = htp.index.reorder_levels([1, 2, 0])
        htp.index.names = ['subbasinID', 'hydrotope', 'time']
        return htp

    @staticmethod
    def from_csv(path, **readkw):
        df = pd.read_csv(path, index_col=[0, 1, 2], parse_dates=[2], **readkw)
        df.index.set_levels(df.index.levels[2].to_period(), 2, inplace=True)
        return df


class hydrotope_daily_crop_indicators(ProjectOrRunData):
    path = osp.join(RESDIR, 'crop.out')
    plugin = []
    column_names = ['doy', 'water_stress', 'temp_stress', 'maturity',
                    'biomass', 'lai', 'root_depth']

    def from_project(self, path, **readkwargs):
        iyr = self.project.config_parameters['iyr']
        df = pd.read_csv(path, delim_whitespace=True, header=None,
                         names=self.column_names)
        prds, hydid = [], []
        idoy, iy, ihyd = 0, iyr-1, 0
        for d in df.pop('doy'):
            ihyd = 1 if d != idoy else ihyd+1
            idoy = d
            hydid += [ihyd]
            iy = iy+1 if d == 1 and ihyd == 1 else iy
            prds += [dt.date(iy, 1, 1)+dt.timedelta(d-1)]
        pix = pd.PeriodIndex(prds, freq='d')
        df.index = pd.MultiIndex.from_arrays([pix, hydid],
                                             names=['time', 'hydrotope'])
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=[0, 1], parse_dates=[0],
                         date_parser=pd.Period, **readkwargs)
        return df


class subbasin_annual_crop_yield(ProjectOrRunData):
    path = osp.join(RESDIR, 'cryld.prn')
    plugin = []
    column_names = ['cropID', 'year', 'subbasinID', 'soilID', 'yield', 'area']
    column_seperators = ['Crp=', 'Yr=', 'Sub=', 'Sol=', 'Yld=', 'Area=']

    def from_project(self, path, **readkwargs):
        df = pd.read_csv(path, sep="|".join(self.column_seperators),
                         engine='python', header=None, names=self.column_names)
        # index
        df.set_index(self.column_names[:4], inplace=True)
        # clean units
        df['yield'] = np.array([y.replace('dt/ha', '') for y in df['yield']],
                               dtype=float)
        df['area'] = np.array([y.replace('ha', '') for y in df['area']],
                              dtype=float)
        return df

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=[0, 1, 2, 3], **readkwargs)
        return df


class gis_files(object):
    """Management plugin to dynamically add GIS file propertyplugins."""

    file_names = {'eva-gis': 'annual_evaporation_actual',
                  'gwr-gis': 'annual_groundwater_recharge',
                  'pre-gis': 'annual_precipitation',
                  'run-gis': 'annual_runoff',
                  }

    class _gis_file(ProjectOrRunData):
        """Generic file interface. path will be assigned through dynamic
        subclassing in gis_files._add_gis_file_propertyplugins.
        """
        plugin = ['to_raster']

        def from_project(self, path, **readkwargs):
            return self.project.gis_files.read(path, **readkwargs)

        def from_csv(self, path, **readkwargs):
            df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
            if len(df.columns) == 1:
                df.index = df.index.astype(int)
                df = df.iloc[:, 0]
            else:
                df.columns = df.columns.astype(int)
                df.index = df.index.to_period()
            return df

        def to_raster(self, timestep=None, prefix=None, name=None, strds=True,
                      mapset=None):
            """Outsourced for reuse to grass.py."""
            _subbasin_or_hydrotope_values_to_raster(
                self.project, self, self.project.hydrotopes.reclass, name=name,
                timestep=timestep, prefix=prefix, strds=strds, mapset=mapset)
            return
        to_raster.__doc__ = _subbasin_or_hydrotope_values_to_raster.__doc__

    def __init__(self, project):
        self.project = project
        self.gisdir = osp.join(project.projectdir, GISDIR)
        self.interfaces = self._create_propertyplugins()
        self.project.settings(**self.interfaces)
        return

    def read(self, pathorname, **readkwargs):
        """Read a SWIM GIS file by full path or by the filename."""
        namepath = osp.join(self.gisdir, pathorname)
        path = namepath if osp.exists(namepath) else pathorname
        df = pd.read_csv(path, delim_whitespace=True, usecols=[0, 2],
                         header=None, names=['id', 'value'], **readkwargs)
        # make 2D array (timesteps, hydrotopes)
        nhyd = df.id.max()
        dfrs = df.value.T.values.reshape(-1, nhyd)
        ids = list(range(1, nhyd+1))
        nsteps = dfrs.shape[0]
        if nsteps > 1:
            ix = self._guess_gis_file_index(nsteps)
            dat = pd.DataFrame(dfrs, columns=ids, index=ix)
        else:
            conf = self.project.config_parameters
            name = '%s:%s' % (conf.start_date, conf.end_date)
            dat = pd.DataFrame(dfrs, columns=ids, index=[name])
        return dat

    def _guess_gis_file_index(self, nsteps):
        nbyr, iyr = self.project.config_parameters('nbyr', 'iyr')
        ixkw = dict(start=str(iyr), periods=nsteps, name='time')
        if nsteps == nbyr:
            ix = pd.period_range(freq='a', **ixkw)
        elif nsteps == nbyr*12:
            ix = pd.period_range(freq='m', **ixkw)
        else:
            ix = pd.period_range(freq='d', **ixkw)
            if ix[-1] != pd.Period(str(iyr+nbyr-1)+'-12-31'):
                msg = 'Last day is %s. Is this really daily?' % ix[-1]
                warnings.warn(msg)
        return ix

    def _create_propertyplugins(self):
        files = glob(osp.join(self.gisdir, '*'))
        plugins = {}
        for f in files:
            class _gf(self._gis_file):
                path = f
            fname = osp.splitext(osp.basename(f))[0]
            altname = fname.replace('-', '_')
            name = 'hydrotope_' + self.file_names.get(fname, altname)
            _gf.__name__ = name
            plugins[name] = propertyplugin(_gf)
        return plugins


# classes attached to project in defaultsettings
PLUGINS = {n: propertyplugin(p) for n, p in globals().items()
           if inspect.isclass(p) and ProjectOrRunData in p.__mro__[1:]}
PLUGINS.update({n: globals()[n] for n in ['gis_files']})
