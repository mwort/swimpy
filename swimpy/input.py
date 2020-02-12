"""
SWIM input functionality.
"""
import os.path as osp
import warnings
import datetime as dt
import inspect

import numpy as np
import pandas as pd
from modelmanager.utils import propertyplugin
from modelmanager.plugins.templates import TemplatesDict
from modelmanager.plugins.pandas import ReadWriteDataFrame
import f90nml

from swimpy import utils, plot
import matplotlib.pyplot as plt  # after plot


class basin_parameters(TemplatesDict):
    """
    Set or get any values from the .bsn file by variable name.
    """
    template_patterns = ['input/*.bsn']
    default_values = {
        'C3C4crop': 0,      'ekc0': 1.0,        'retPsur': 20.0,
        'CO2ref': 0,        'epco': 1.0,        'rnew': 0.08,
        'CO2scen': 0,       'evrch': 1.0,       'roc2': 5.0,
        'abf0': 0.0,        'gmrate': 10.0,     'roc4': 5.0,
        'bDormancy': 0,     'gwq0': 0.0,        'sccor': 1.0,
        'bResModule': 0,    'ialpha': 0,        'smrate': 0.5,
        'bRunoffDat': 0,    'ibeta': 0,         'snow1': 0.0,
        'bSnowModule': 1,   'icn': 0,           'spcon': 0.0,
        'bff': 1.0,         'idlef': 0,         'spexp': 1.0,
        'chcc0': 1.0,       'idvwk': 0,         'stinco': 0.0,
        'chnnc0': 1.0,      'iemeth': 0,        'storc1': 0.0,
        'chwc0': 1.0,       'intercep': 1,      'subcatch': 0,
        'chxkc0': 1.0,      'isc': 0,           'tgrad1': -0.0068,
        'cnum1': 1.0,                           'thc': 1.0,
        'cnum2': 1.0,       'maxup': 0.0,       'tlgw': 0,
        'cnum3': 1.0,       'prcor': 1.0,       'tlrch': 1.0,
        'degNgrw': 0.3,     'prf': 1.0,         'tmelt': 0.0,
        'degNsub': 0.3,     'radiation': 0,     'tsnfall': 0.0,
        'degNsur': 0.02,    'rdcor': 1.0,       'ulmax0': 1.0,
        'degPsur': 0.02,    'retNgrw': 15000.0, 'xgrad1': 0.0,
        'ec1': 0.135,       'retNsub': 365.0,
        'ecal': 1.0,        'retNsur': 5.0,
        }

    def set_default(self, *subset, **override):
        """Set the basin parameters to neutral or standard values."""
        pn = subset or self.default_values.keys()
        new = {i: self.default_values[i] for i in pn}
        new.update(override)
        self(**new)
        return


class config_parameters(TemplatesDict):
    """
    Set or get any values from the .cod or swim.conf file by variable name.
    """
    template_patterns = ['input/*.cod', 'swim.conf']

    @property
    def start_date(self):
        return dt.date(self['iyr'], 1, 1)

    @property
    def end_date(self):
        return dt.date(self['iyr']+self['nbyr']-1, 12, 31)

    def __getitem__(self, k):
        v = TemplatesDict.__getitem__(self, k)
        path = osp.abspath(osp.join(self.project.projectdir, str(v)))
        return path if osp.exists(path) else v


class subcatch_parameters(ReadWriteDataFrame):
    """
    Read or write parameters in the subcatch.prm file.
    """
    path = 'input/subcatch.prm'
    index_name = 'catchmentID'
    force_dtype = {index_name: int}

    def read(self, **kwargs):
        bsn = pd.read_csv(self.path, delim_whitespace=True,
                          dtype=self.force_dtype)
        stn = 'stationID' if 'stationID' in bsn.columns else 'station'
        bsn.set_index(stn, inplace=True)
        return bsn

    def write(self, **kwargs):
        # make sure catchmentID is first column
        if self.columns[0] != self.index_name:
            if self.index_name in self.columns:
                cid = self.pop(self.index_name)
            else:
                cid = self.project.stations.loc[self.index, 'stationID']
            self.insert(0, self.index_name, cid)
        bsn = self.sort_values(self.index_name)
        bsn['stationID'] = bsn.index
        strtbl = bsn.to_string(index=False, index_names=False)
        with open(self.path, 'w') as f:
            f.write(strtbl)
        return


class subcatch_definition(ReadWriteDataFrame):
    """
    Interface to the subcatchment definition file from DataFrame or grass.
    """
    path = 'input/subcatch.def'
    plugin = ['update']

    def read(self, **kwargs):
        scdef = pd.read_csv(self.path, delim_whitespace=True, index_col=0)
        return scdef

    def write(self, **kwargs):
        tbl = self.copy()
        tbl.insert(0, 'subbasinID', tbl.index)
        tblstr = tbl.to_string(index=False, index_names=False)
        with open(self.path, 'w') as f:
            f.write(tblstr)
        return

    def update(self, catchments=None, subbasins=None):
        """Write the definition file from the subbasins grass table.

        Arguments
        ---------
        catchments : list-like
            Catchment ids to subset the table to. Takes precedence over
            subbasins argument.
        subbasins : list-like
            Subbasin ids to subset the table to.
        """
        from modelmanager.plugins.grass import GrassAttributeTable

        cols = ['subbasinID', 'catchmentID']
        tbl = GrassAttributeTable(self.project, subset_columns=cols,
                                  vector=self.project.subbasins.vector)
        # optionally filter
        if catchments is not None:
            tbl = tbl[[i in catchments for i in tbl.catchmentID]]
        elif subbasins is not None:
            tbl = tbl.filter(items=subbasins, axis=0)
        # add stationID
        scp = {v: k for k, v in
               self.project.subcatch_parameters['catchmentID'].items()}
        tbl['stationID'] = [scp[i] for i in tbl['catchmentID']]
        # save and write
        self.__call__(tbl)
        return

    def subcatch_subbasin_ids(self, catchmentID):
        """Return all subbasinIDs of the subcatchment."""
        return self.index[self.catchmentID == catchmentID].values

    def catchment_subbasin_ids(self, catchmentID):
        """Return all subbasins of the catchment respecting the topology.

        The `project.stations` "ds_stationID" column needs to give the from-to
        topology of catchments/stations.
        """
        ft = self.project.stations['ds_stationID']
        all_catchments = [catchmentID] + utils.upstream_ids(catchmentID, ft)
        ssid = self.subcatch_subbasin_ids
        return np.concatenate([ssid(i) for i in all_catchments])


class station_output(ReadWriteDataFrame):
    """
    Interface to the station output file.
    """
    path = 'input/gauges.output'
    plugin = ['update']

    def read(self, **kwargs):
        scdef = pd.read_csv(self.path, delim_whitespace=True, index_col=1)
        return scdef

    def write(self, **kwargs):
        tbl = self.copy()
        tbl.insert(1, 'stationID', tbl.index)
        tblstr = tbl.to_string(index=False, index_names=False)
        with open(self.path, 'w') as f:
            f.write(tblstr)
        return

    def update(self, stations=None):
        """Write the definition file from project.stations table.

        Arguments
        ---------
        stations : list-like
            Station ids to subset the table to. Default is all stations.
        """
        t = self.project.stations.loc[stations or slice(None), ['subbasinID']]
        # save and write
        self.__call__(t)
        return


class climate(object):
    """All climate input related functionality."""

    def __init__(self, project):
        self.project = project
        return

    @propertyplugin
    class inputdata(ReadWriteDataFrame):
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
            ReadWriteDataFrame.__init__(self, project)
            return

        def read(self, climdir=None, **kw):
            startyr = self.project.config_parameters['iyr']
            path = osp.join(climdir or self.path, self.namepattern)
            dfs = pd.concat([self.read_clim(path % i, startyr, vs, **kw)
                             for i, vs in self.clim_variables.items()], axis=1)
            dfs.sort_index(axis=1, inplace=True)
            return dfs

        @classmethod
        def read_clim(cls, path, startyear, variables, **readkwargs):
            """Read single clim file and return DataFrame with index and
            columns.
            """
            assert len(variables) == 3
            readargs = dict(delim_whitespace=True, header=None, skiprows=1)
            readargs.update(readkwargs)
            df = pd.read_csv(path, **readargs)
            df.index = pd.period_range(start=str(startyear), periods=len(df),
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
        def plot_temperature(self, regime=False, freq='d', minmax=True,
                             ax=None, runs=None, output=None, **linekw):
            """Line plot of mean catchment temperature.

            Arguments
            ---------
            regime : bool
                Plot regime. freq must be 'd' or 'm'.
            freq : <pandas frequency>
                Any pandas frequency to aggregate to.
            minmax : bool
                Show min-max range.
            **kw :
                Parse any keyword to the tmean line plot function.
            """
            ax = ax or plt.gca()
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
        def plot_precipitation(self, regime=False, freq='d',
                               ax=None, runs=None, output=None, **barkwargs):
            """Bar plot of mean catchment precipitation.

            Arguments
            ---------
            regime : bool
                Plot regime. freq must be 'd' or 'm'.
            freq : <pandas frequency>
                Any pandas frequency to aggregate to.
            **barkwargs :
                Parse any keyword to the bar plot function.
            """
            ax = ax or plt.gca()
            clim = self.aggregate(variables=['precipitation'],
                                  freq=freq, regime=regime)['precipitation']
            bars = plot.plot_precipitation_bars(clim, ax, **barkwargs)
            if regime:
                xlabs = {'d': 'Day of year', 'm': 'Month'}
                ax.set_xlabel(xlabs[freq])
            return bars

    @propertyplugin
    class config_parameters(f90nml.Namelist):
        path = 'ncinfo.nml'
        _nml = None
        plugin = ['__call__']

        def __init__(self, project):
            self.path = osp.join(project.projectdir,
                                 project.config_parameters['climatedir'],
                                 self.path)
            f90nml.Namelist.__init__(self)
            nml = f90nml.read(self.path)
            self.update(nml['nc_parameters'])
            self._nml = nml
            return

        def write(self, path=None):
            self._nml["nc_parameters"].update(self)
            self._nml.write(path or self.path, force=True)
            return

        def __setitem__(self, key, value):
            f90nml.Namelist.__setitem__(self, key, value)
            if self._nml:
                self.write()
            return

        def __call__(self, *get, **set):
            assert get or set
            if set:
                self.update(set)
            if get:
                return [self[k] for k in get]
            return


class structure_file(ReadWriteDataFrame):
    """Read-Write plugin for the structure file.

    This is accessible via the ``hydroptes.attributes`` propertyplugin and
    placed here for consistency and reuse.
    """
    file_columns = ['subbasinID', 'landuseID', 'soilID', 'management',
                    'wetland', 'elevation', 'glacier', 'area', 'cells',
                    'irrigation']

    @property
    def path(self):
        relpath = 'input/%s.str' % self.project.project_name
        return self._path if hasattr(self, '_path') else relpath

    @path.setter
    def path(self, value):
        self._path = value
        return

    def read(self, **kwargs):
        df = pd.read_csv(self.path, delim_whitespace=True)
        # pandas issues UserWarning if attribute is set with Series-like
        warnings.simplefilter('ignore', UserWarning)
        self.file_header = list(df.columns)
        warnings.resetwarnings()

        nstr, nexp = len(df.columns), len(self.file_columns)
        if nexp == nstr:
            df.columns = self.file_columns
        else:
            msg = ('Non-standard column names: Found different number of '
                   'columns in .str file, expecting %i, got %i: %s')
            warnings.warn(msg % (nexp, nstr, ', '.join(df.columns)))
        # get rid of last 0 line
        if df.iloc[-1, :].sum() == 0:
            df = df.iloc[:-1, :]
        df.index = list(range(1, len(df)+1))
        return df

    def write(self, **kwargs):
        with open(self.path, 'w') as f:
            self.to_string(f, index=False, header=self.file_header)
            f.write('\n'+' '.join(['0 ']*len(self.file_header)))
        return


class station_daily_discharge_observed(ReadWriteDataFrame):
    path = 'input/runoff.dat'
    subbasins = []  #: Holds subbasinIDs if the file has them
    outlet_station = None  #: Name of the first column which is always written

    def read(self, path=None, **kwargs):
        path = path or self.path
        na_values = ['NA', 'NaN', -999, -999.9, -9999]
        # first read header
        with open(path, 'r') as fi:
            colnames = fi.readline().strip().split()
            subids = fi.readline().strip().split()
        skiphead = 1
        # subbasins are given if all are ints and they are all in the subbasins
        try:
            si = pd.Series(subids, dtype=int, index=colnames)
            if list(si.iloc[1:3]) == [0, 0]:
                self.subbasins = si.iloc[3:]
                skiphead += 1
        except ValueError:
            warnings.warn('No subbasinIDs given in second row of %s' % path)
        # read entire file
        rodata = pd.read_csv(path, skiprows=skiphead, header=None, index_col=0,
                             delim_whitespace=True, parse_dates=[[0, 1, 2]],
                             names=colnames, na_values=na_values)
        rodata.index = rodata.index.to_period()
        self.outlet_station = rodata.columns[0]
        return rodata

    def write(self, **kwargs):
        head = 'YYYY  MM  DD  ' + '  '.join(self.columns.astype(str)) + '\n'
        if len(self.subbasins) > 0:
            sbids = '  '.join(self.subbasins.astype(str))
            head += '%s  0  0  ' % len(self.columns) + sbids + '\n'
        # write out
        out = [self.index.year, self.index.month, self.index.day]
        out += [self[s] for s in self.columns]
        out = pd.DataFrame(list(zip(*out)))
        with open(self.path, 'w') as fo:
            fo.write(head)
            out.to_string(fo, na_rep='-9999', header=False, index=False)
        return

    def __call__(self, data=None, stations=[], start=None, end=None):
        """Write daily_discharge_observed from stations with their subbasinIDs.

        Arguments
        ---------
        data : pd.DataFrame
            DataFrame to write with stationID columns. Takes presence over
            stations and may or may not include outlet_station.
        stations : list-like, optional
            Stations to write to file. self.outlet_station will always be
            written as the first column.
        start, end : datetime-like, optional
            Start and end to write to. Defaults to
            project.config_parameters.start_date/end_date.
        """
        if data is None:
            data = self._get_observed_discharge(stations=stations, start=start,
                                                end=end)
        elif self.outlet_station not in data.columns:
            start, end = data.index[[0, -1]].astype(str)
            osq = self._get_observed_discharge(start=start, end=end)
            data.insert(0, self.outlet_station, osq[self.outlet_station])

        # update subbasins
        self.subbasins = self.project.stations.loc[data.columns, 'subbasinID']
        # assign to self
        pd.DataFrame.__init__(self, data)
        self.write()
        return self

    def _get_observed_discharge(self, stations=[], start=None, end=None):
        """Get daily_discharge_observed from stations and their subbasinIDs.

        Arguments
        ---------
        stations : list-like, optional
            Stations to write to file. self.outlet_station will always be
            written as the first column.
        start, end : datetime-like, optional
            Start and end to write to. Defaults to
            project.config_parameters.start_date/end_date.
        """
        stat = [self.outlet_station]
        stat += [s for s in stations if s != self.outlet_station]
        # unpack series from dataframe, in right order!
        stationsq = self.project.stations.daily_discharge_observed
        si = [s for s in stat if s not in stationsq.columns]
        assert not si, ('%s not found in stations.daily_discharge_observed: %s'
                        % (si, stationsq.columns))
        q = stationsq[stat]
        # change start/end
        conf = self.project.config_parameters
        q = q.truncate(before=start or conf.start_date,
                       after=end or conf.end_date)
        return q


# classes attached to project in defaultsettings
PLUGINS = {n: propertyplugin(p) for n, p in globals().items()
           if inspect.isclass(p) and
           set([ReadWriteDataFrame, TemplatesDict]) & set(p.__mro__[1:])}
PLUGINS.update({n: globals()[n] for n in ['climate']})
