"""
SWIM input functionality with interface to GRASS.
"""
import os.path as osp
from warnings import warn
import datetime as dt
import subprocess
import inspect

import numpy as np
import pandas as pd
import f90nml
from modelmanager.utils import propertyplugin
from modelmanager.plugins.templates import TemplatesDict
from modelmanager.plugins.pandas import ReadWriteDataFrame
from modelmanager.plugins import grass as mmgrass

from swimpy import utils, plot
from swimpy.grass import reclass_raster, get_modulearg
import matplotlib.pyplot as plt  # after plot


class ParamGroupNamelist(f90nml.Namelist):
    """
    Namelist class for individual parameter groups (project.<group>_parameters).
    """
    _defaults = None

    def __init__(self, nml, pargrp, project):
        self.project = project
        self._pargrp = pargrp
        self._defaults = self.defaults
        super().__init__(nml)
    
    def __getitem__(self, key):
        # return default value if not found
        if not key in self.keys():
            if key not in self.defaults.keys():
                raise KeyError("Parameter '{}' not implemented!".format(key))
            return self.defaults[key]
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if not key in self.defaults.keys():
            raise KeyError("Parameter '{}' not implemented!".format(key))
        return super().__setitem__(key, value)

    @property
    def defaults(self):
        """
        Show all implemented parameters with their default values.
        """
        if self._defaults is None:
            nml = f90nml.reads(subprocess.check_output([self.project.swim, "-d"]).decode())
            nl = nml[self._pargrp]
            # trim whitespaces
            for k, v in nl.items():
                if isinstance(v, str):
                    nl[k] = v.strip()
            self._defaults = nl
        return self._defaults
    
    def __call__(self, *get, **set):
        assert get or set
        if set:
            self.update(set)
            self.project.config_parameters.write()
        if get:
            return [self[k] for k in get]
        return
    
    def set_default(self, *pars):
        """Set parameter(s) to default value(s)."""    
        new = {p: self.defaults[p] for p in pars}
        self(**new)
        return


class config_parameters(f90nml.Namelist):
    """
    Set or get values from <config>.nml. For all parameters not in the list,
    default values are used.
    """
    _defaults = None

    def __init__(self, project):
        self.project = project
        nml = f90nml.read(osp.join(self.project.projectdir,
                                   self.project.parfile))
        super().__init__(nml)
        # empty Namelist for groups not in .nml file
        mis = {k: f90nml.Namelist() for k in self.defaults.keys() if k not in self.keys()}
        if len(mis) > 0:
            self(**mis)
        # namelist groups (<group>_parameters) as project attributes
        for k, nl in self.items():
            self[k] = ParamGroupNamelist(nl, k, self.project)
            setattr(self.project, k, self[k])
        return
    
    @property
    def defaults(self):
        """
        Show all implemented parameters with their default values.
        """
        if self._defaults is None:
            nml = f90nml.reads(subprocess.check_output([self.project.swim, "-d"]).decode())
            # trim whitespaces
            for gr, nl in nml.items():
                for k, v in nl.items():
                    if isinstance(v, str):
                        nml[gr][k] = v.strip()
            self._defaults = nml
        return self._defaults

    def __getitem__(self, key):
        # return default value if not found
        if not key in self.keys():
            for gr, nl in self.defaults.items():
                if key in nl:
                    return self[gr][key]
            raise KeyError("Parameter or parameter group '{}' not implemented!".format(key))
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        if not key in self.defaults.keys():
            for gr, nl in self.defaults.items():
                if key in nl:
                    return self[gr].__setitem__(key, value)
            raise KeyError("Parameter or parameter group '{}' not implemented!".format(key))
        return super().__setitem__(key, value)
    
    @property
    def parlist(self):
        """
        Show all parameters with their values without distinction of parameter groups. 
        """
        return self._parlist(self)

    @staticmethod
    def _parlist(pars):
        return {k: v for p in pars.values() if len(p) > 0 for k, v in p.items()}
    
    def __call__(self, *get, **set):
        assert get or set
        if set:
            for k, v in set.items():
                # k is a parameter group
                if k in self.defaults.keys():
                    self[k] = ParamGroupNamelist(v, k, self.project)
                    setattr(self.project, k, self[k])
                # k is a specific parameter
                else:
                    self.__setitem__(k, v)
                self.write()
        if get:
            return [self[k] for k in get]
        return
    
    def set_default(self, *pars):
        """Set parameter group(s) or individual parameter(s) to default values."""
        new = {}
        for p in pars:
            if p in self.defaults.keys():
                new.update({p: self.defaults[p]})
            elif p in self._parlist(self.defaults).keys():
                override = {p: v[p] for v in self.defaults.values() if p in v.keys()}
                new.update(override)
            else:
                raise KeyError("Parameter or parameter group '{}' not implemented!".format(p))
        self(**new)
        return
    
    def write(self, file=None):
        """
        Write current parameters into project.parfile or a new .nml file.

        Args:
            file: (Optional) A file name into which parameters are written.
        """
        path = file or osp.join(self.project.projectdir,
                                self.project.parfile)
        return super().write(path, force=True)

    @property
    def start_date(self):
        return dt.date(self('iyr')[0], 1, 1)

    @property
    def end_date(self):
        return dt.date(self('iyr')[0]+self('nbyr')[0]-1, 12, 31)

    def __repr__(self):
        rpr = '<{}: {}>\n'.format(self.__class__.__name__, self.project.parfile)
        pargrp = 'f90nml.Namelist (\n'
        for k, v in self.items():
            ne, le = 5, len(v)
            v_sel = list(v.items())[:min(ne, le)]
            entries = ['{}: {}'.format(vk, vv) for (vk, vv) in v_sel]
            cont = (', ... {} more'.format(le - ne) if le > ne else '')
            pargrp += k + ': Namelist (' + ', '.join(entries) + cont + ')\n'
        return rpr + pargrp + ')'


class InputFile(ReadWriteDataFrame):
    """
    Abstract class for generic input file handling. No project attribute.
    """
    
    @property
    def path(self):
        if self.file:
            relpath = osp.join(self.project.inputpath, self.file)
        else:
            relpath = None
        return self._path if hasattr(self, '_path') else relpath

    @path.setter
    def path(self, value):
        self._path = value
        return

    def read(self, **kwargs):
        """Read input file. Result object inherits from pandas.DataFrame. 

        Arguments
        ---------
        **kwargs :
            Keywords to pandas.read_csv.
        """
        if self.path:
            na_values = ['NA', 'NaN', -999, -999.9, -9999]
            df = pd.read_csv(self.path, skipinitialspace=True,
                            index_col=self.index_name, na_values=na_values,
                            **kwargs)
            df.columns = df.columns.str.strip()
        else:
            df = None
        return df

    def write(self, **kwargs):
        """Write to csv file.

        Arguments
        ---------
        **kwargs :
            Keywords to pandas.to_csv.
        """
        if self.path:
            self.to_csv(self.path, index = True, na_rep='-9999', **kwargs)
        else:
            warn("No file has been given, nothing to do!")
        return


class InputFileGrassModule(InputFile, mmgrass.GrassModulePlugin):
    """Abstract class for <InputFile>s that are related to m.swim GRASS modules."""

    def __init__(self, projectorpath, read=True, **kwargs):
        try:
            InputFile.__init__(self, projectorpath, read, **kwargs)
        except AssertionError as e:
            InputFile.__init__(self, projectorpath, read=False, **kwargs)
            warn(f'{e}. Try to run update() method to (re-)create the input file!')
        mmgrass.GrassModulePlugin.__init__(self, projectorpath)
        return

    def create(self, **modulekwargs):
        """Run specific m.swim.* GRASS module.
        
        Arguments
        ---------
            **modulekwargs :
                Parameters of m.swim.* (overrides or adds to 'grass_setup'
                from settings.py).
        """
        if 'output' not in modulekwargs:
            modulekwargs['output'] = self.path
        return mmgrass.GrassModulePlugin.create(self, **modulekwargs)
    
    def update(self, **modulekwargs):
        """Calls create() and postprocess() methods. Basically running the
        required m.swim.* module(s).
        
        Arguments
        ---------
            **modulekwargs :
                Parameters of m.swim.* (overrides or adds to 'grass_setup'
                from settings.py).
        """
        return mmgrass.GrassModulePlugin.update(self, **modulekwargs)

    def reclass(self, values, outrast, mapset=None):
        """Reclass corresponding GRASS raster to 'values'.

        Arguments
        ---------
        values : list-like | <pandas.Series>
            Values the raster is reclassed to. If a pandas.Series is parsed,
            the index is used to create the mapping, otherwise the categories
            are assumed to be ``range(1, len(values)+1)``.
        outrast : str
            Name of raster to be created.
        mapset : str, optional
            Different than the default `grass_mapset` mapset to write to.
        """
        if not hasattr(self, 'raster'):
            raise NotImplementedError('No reclass method implemented for that attribute!')
        hydname = self.raster + '@' + self.project.grass_mapset
        reclass_raster(self.project, hydname, outrast, values, mapset=mapset)
        return


class hydrotope(InputFileGrassModule):
    """
    Read or write hydrotope.csv and interface to related GRASS module m.swim.hydrotopes.
    """
    file = 'hydrotope.csv'
    index_name = 'hydrotope_id'
    # grass module
    argument_setting = 'grass_setup'
    module = 'm.swim.hydrotopes'
    # other class variables
    raster = property(lambda self: get_modulearg(self.project, self.module, 'hydrotopes'))

class structure_file(hydrotope):

    def __init__(self, *args, **kwargs):
        warn(f'{self.__class__.__name__} is deprecated and will be removed in '
             'a future version. Use project.hydrotope instead.',
             FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class subbasin_routing(InputFileGrassModule):
    """
    Read or write subbasin_routing.csv and interface to related GRASS module m.swim.routing.
    """
    file = 'subbasin_routing.csv'
    index_name = None
    # grass module
    argument_setting = 'grass_setup'
    module = 'm.swim.routing'


class subbasin(InputFileGrassModule):
    """
    Read or write subbasin.csv and interface to related GRASS modules m.swim.*.
    """
    file = 'subbasin.csv'
    index_name = 'subbasin_id'
    # grass module arguments as class variables
    argument_setting = 'grass_setup'
    module = 'm.swim.subbasins'
    # other class variables
    vector = property(lambda self: get_modulearg(self.project, self.module, 'subbasins'))
    raster = vector

    @propertyplugin
    class substats(InputFileGrassModule):
        """Interface to GRASS module m.swim.substats"""
        file = 'subbasin.csv'
        index_name = 'subbasin_id'
        argument_setting = 'grass_setup'
        module = 'm.swim.substats'

        def __call__(self, **modulekwargs):
            """Run GRASS module m.swim.substats."""
            return self.update(**modulekwargs)

    @propertyplugin
    class attributes(mmgrass.GrassAttributeTable):
        """The subbasins attribute table as a ``pandas.DataFrame`` object."""
        vector = property(lambda self: self.project.subbasin.vector)

    def postprocess(self, **moduleargs):
        self.project.subbasin_routing.update(**moduleargs)
        self.project.subbasin.substats(**moduleargs)
        self.project.hydrotope.update(**moduleargs)
        self.project.catchment.update()
        # TODO: write nc_climate files
        return


class catchment(InputFileGrassModule):
    """
    Read or write catchment.csv
    """
    file = 'catchment.csv'
    index_name = 'station_id'
    # grass module arguments as class variables
    argument_setting = 'grass_setup'
    module = 'm.swim.subbasins'
    # other class variables
    raster = property(lambda self: get_modulearg(self.project, self.module, 'catchments'))
    vector = raster
    
    @propertyplugin
    class attributes(mmgrass.GrassAttributeTable):
        """The subbasins attribute table as a ``pandas.DataFrame`` object."""
        vector = property(lambda self: self.project.catchment.vector)

    def update(self, catchments=None, subbasins=None):
        """Update catchment.csv from the subbasins grass table and
        stations from settings.py.

        WARNING: default parameters from project.catchment_defaults will be
        assigned, manual changes to catchment.csv will be lost!

        Arguments
        ---------
        catchments : list-like
            Catchment ids to subset the table to. Takes precedence over
            subbasins argument.
        subbasins : list-like
            Subbasin ids to subset the table to.
        """

        tbl = self.project.subbasin.attributes
        # optionally filter
        if catchments is not None:
            tbl = tbl[[i in catchments for i in tbl.catchmentID]]
        elif subbasins is not None:
            tbl = tbl.filter(items=subbasins, axis=0)
        # join station names
        tbl.set_index('catchmentID', inplace=True)
        scp = self.project.stations
        scp.reset_index(inplace=True)
        scp.set_index('stationID', inplace=True)
        catch = tbl.join(scp, rsuffix = '_t')
        # select and rename columns
        catch = catch['NAME']
        catch.index.name = 'catchment_id'
        catch.name = 'station_id'
        # unique catchments
        catch.drop_duplicates(inplace=True)
        catch = catch.to_frame()
        catch.reset_index(inplace=True)
        # add default parameters
        pdf = pd.DataFrame(self.project.catchment_defaults, index=[0])
        pdf = pd.concat([pdf]*len(catch), ignore_index=True)
        out = pd.concat([catch, pdf], axis = 1)
        out.set_index('station_id', inplace=True)
        # save and write
        self.__call__(out)
        return

    def subcatch_subbasin_ids(self, catchmentID):
        """Return all subbasinIDs of the subcatchment."""
        return self.project.subbasin.loc[lambda df: df['catchment_id'] == catchmentID].index.values

    def catchment_subbasin_ids(self, catchmentID):
        """Return all subbasins of the catchment respecting the topology.

        The `project.stations` "ds_stationID" column needs to give the from-to
        topology of catchments/stations.
        """
        ft = self.project.stations['ds_stationID']
        all_catchments = [catchmentID] + utils.upstream_ids(catchmentID, ft)
        ssid = self.subcatch_subbasin_ids
        return np.concatenate([ssid(i) for i in all_catchments])

class subcatch_parameters(catchment):

    def __init__(self, *args, **kwargs):
        warn(f'{self.__class__.__name__} is deprecated and will be removed in '
             'a future version. Use project.catchment instead.',
             FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

class subcatch_definition(catchment):
    def __init__(self, *args, **kwargs):
        warn(f'{self.__class__.__name__} is deprecated and will be removed in '
             'a future version. Use project.catchment instead.',
             FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class climate(object):
    """
    All climate input related functionality.
    
    TODO: Give warning if both climate.csv and netcdf climate input are found
    """

    def __init__(self, project):
        self.project = project
        return

    @propertyplugin
    class inputdata(InputFile):
        """A lazy DataFrame representation of climate.csv.

        Rather than being read on instantiation, .read() and .write() need to
        be called explicitly since both operations are commonly time-consuming.
        """
        file = 'climate.csv'
        plugin = ['print_stats', 'plot_temperature', 'plot_precipitation']

        def read(self):
            df = pd.read_csv(self.path, skipinitialspace=True,
                             parse_dates=['time'], index_col='time')
            df.columns = df.columns.str.strip()
            # multi-index columns
            df = pd.pivot_table(df, index='time', columns=['subbasin_id'])
            df.columns.names = ['variable', 'subbasin_id']
            return df

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
            vars = variables or self.columns.levels[0].tolist()
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
    class netcdf_inputdata(object):
        variables = ['tmean', 'tmin', 'tmax', 'precipitation',
                     'radiation', 'humidity']

        def __init__(self, project):
            self.project = project
            self.path = project.inputpath
            self.parameters = project.nc_climate_parameters
            return

        @property
        def grid_mapping(self):
            pth = osp.join(self.path,
                           self.parameters["nc_grid"])
            grid = pd.read_csv(pth, skipinitialspace=True,
                               index_col='subbasinID')
            grid.columns = grid.columns.str.strip()
            return grid

        def read_gridded(self, variable, time=None, subbasins=None):
            """Read a variable from the netcdf files and return as grid.

            Arguments
            ---------
            variable : str
                Qualified name of climate variable (cf. self.variables)
            time : <any pandas time index> | (from, to), optional
                A time slice to read. Use a tuple of (from, to) for slicing,
                including None for open sides. Default: read all.
            subbasins : list-like, optional
                Only read for a subset of subbasins.
            """
            import netCDF4 as nc
            msg = variable+" not in %r" % self.variables
            assert variable in self.variables, msg
            vfn = zip(self.parameters["nc_vnames"], self.parameters["nc_fnames"])
            vname, file_path = dict(zip(self.variables, vfn))[variable]
            ds = nc.Dataset(osp.join(self.path, file_path))
            # get space indeces
            lon = ds[self.parameters["nc_lon_vname"]][:]
            lonix = pd.Series(range(len(lon)), index=lon)
            lat = ds[self.parameters["nc_lat_vname"]][:]
            latix = pd.Series(range(len(lat)), index=lat)
            grid = self.grid_mapping
            if subbasins:
                grid = grid.loc[subbasins]
            lons = lonix[grid.lon.unique()].sort_values()
            lats = latix[grid.lat.unique()].sort_values()
            cl = pd.MultiIndex.from_product((lats.index, lons.index))
            # get space indeces
            st = pd.Period(str(self.parameters["nc_ref_year"]), freq="d")
            timeint = np.array(ds[self.parameters["nc_time_vname"]][:], dtype=int)
            pix = st + self.parameters["nc_offset_days"] + timeint
            tix = pd.Series(range(len(timeint)), index=pix)
            # get time indeces
            if time and type(time) == tuple and len(time) == 2:
                tix = tix[time[0]:time[1]]
            elif time:
                tix = tix[time]
            # read data
            data = (ds[vname][tix.values, lats.values, lons.values]
                    .reshape(-1, len(cl)))
            df = pd.DataFrame(data, columns=cl, index=tix.index)
            ds.close()
            return df

        def read(self, variable, time=None, subbasins=None):
            """Return climate variable as subbasin weighted means."""
            grid = self.grid_mapping
            if subbasins:
                grid = grid.loc[subbasins]
            ggb = grid.groupby(grid.index)
            gridded = self.read_gridded(variable, time=time)
            dt = gridded.dtypes.mode()[0]
            # single value and weighted means separately
            # (trade-off btw speed & memory)
            cnt = ggb.weight.count()
            data = pd.DataFrame(
                dtype=dt, columns=cnt.index, index=gridded.index)
            c1 = cnt[cnt == 1].index
            cix = [(la, lo) for la, lo in grid.loc[c1, ["lat", "lon"]].values]
            data[c1] = gridded[cix].values
            # weighted means looped to avoid large copied array
            for s in cnt[cnt > 1].index:
                c = [(la, lo) for la, lo in grid.loc[s, ["lat", "lon"]].values]
                wght = grid.loc[s, 'weight']/grid.loc[s, 'weight'].sum()
                data[s] = gridded[c].mul(wght.values).sum(axis=1).astype(dt)
            return data
        read.__doc__ += read_gridded.__doc__[58:]

        def __getitem__(self, key):
            if type(key) == str:
                return self.read(key)
            elif hasattr(key, "__iter__"):
                return pd.concat([self.read(k) for k in key], axis=1, keys=key)
            else:
                raise KeyError("%s not in %r" % (key, self.variables))


class discharge(InputFile):
    """
    Read and write discharge.csv.
    """
    file = 'discharge.csv'
    index_name = 'time'
    subbasins = []  #: Holds subbasinIDs if the file has them
    outlet_station = None  #: Name of the first column which is always written

    @property
    def outlet_station(self):
        stat = self.columns[0]
        return self._outlet_station if hasattr(self, '_outlet_station') else stat

    @outlet_station.setter
    def outlet_station(self, value):
        self._outlet_station = value
        return
    
    def read(self, **kwargs):
        df = super().read(parse_dates=True, **kwargs)
        df.index = df.index.to_period()
        return df

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

class station_daily_discharge_observed(discharge):

    def __init__(self, *args, **kwargs):
        warn(f'{self.__class__.__name__} is deprecated and will be removed in '
             'a future version. Use project.discharge instead.',
             FutureWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


# class station_output(ReadWriteDataFrame):
#     """
#     Interface to the station output file.
#     """
#     path = 'input/gauges.output'
#     plugin = ['update']

#     def read(self, **kwargs):
#         scdef = pd.read_csv(self.path, delim_whitespace=True, index_col=1)
#         return scdef

#     def write(self, **kwargs):
#         tbl = self.copy()
#         tbl.insert(1, 'stationID', tbl.index)
#         tblstr = tbl.to_string(index=False, index_names=False)
#         with open(self.path, 'w') as f:
#             f.write(tblstr)
#         return

#     def update(self, stations=None):
#         """Write the definition file from project.stations table.

#         Arguments
#         ---------
#         stations : list-like
#             Station ids to subset the table to. Default is all stations.
#         """
#         t = self.project.stations.loc[stations or slice(None), ['subbasinID']]
#         # save and write
#         self.__call__(t)
#         return


# classes attached to project in defaultsettings
PLUGINS = {n: propertyplugin(p) for n, p in globals().items()
           if inspect.isclass(p) and
           n not in ['InputFile', 'InputFileGrassModule'] and
           set([ReadWriteDataFrame, TemplatesDict]) & set(p.__mro__[1:])}
PLUGINS.update({n: globals()[n] for n in ['climate', 'config_parameters']})
