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
import itertools
import warnings
# from glob import glob
# import datetime as dt
# import calendar
# import inspect

# import numpy as np
import pandas as pd
import f90nml
import matplotlib.pyplot as plt
from matplotlib import cm
from modelmanager.utils import propertyplugin
from modelmanager.plugins.pandas import ProjectOrRunData

from swimpy import utils, hydro, plot
#from swimpy.plot import plot_function as _plot_function
from swimpy.grass import _to_raster


class OutputFile(ProjectOrRunData):
    """
    Abstract class for generic output file handling.
    """
    file = None

    def __init__(self, projectrunorpath):
        super().__init__(projectrunorpath)

        # dynamically add further postprocessing methods
        if self._exists:
            vars = self.columns.get_level_values('variable').unique()
            stats = self.columns.get_level_values(self._space).unique()

        # discharge methods
        def obs_sim_overlap(warmupyears=1):
            """Return overlapping obs and sim dataframes excluding warmup period.

            Arguments
            ---------
            warmupyears : int
                Number of years to skip at beginng as warm up period.

            Returns
            -------
            (pd.DataFrame, pd.DataFrame) : observed and simulated discharge.
            """
            obs = getattr(self.project.stations, self._time + '_discharge_observed')
            # exclude warmup period
            sim = self[str(self.index[0].year+warmupyears):]['discharge']
            obsa, sima = obs.align(sim, join='inner')
            # obsa can still have columns with only NAs
            obsa.dropna(how='all', axis=1, inplace=True)
            simout = sima[obsa.columns]
            # give warning if empty
            if obsa.empty or simout.empty:
                warnings.warn('Simulation and observation data do not overlap, discharge metrics are empty!')
            return obsa, simout

        def NSE():
            """pandas.Series of Nash-Sutcliff efficiency excluding warmup year."""
            obs, sim = self.obs_sim_overlap()
            return pd.Series({s: hydro.NSE(obs[s], sim[s]) for s in obs.columns})

        def rNSE():
            """pandas.Series of reverse Nash-Sutcliff efficiency (best = 0)"""
            return 1 - self.NSE

        def pbias():
            """pandas.Series of percent bias excluding warmup year."""
            obs, sim = self.obs_sim_overlap()
            return pd.Series({s: hydro.pbias(obs[s], sim[s]) for s in obs.columns})

        def pbias_abs():
            """pandas.Series of absolute percent bias excluding warmup year."""
            return self.pbias.abs()

        def plot_discharge_comparison(freq=None, **plotkw):
            """Line plot of discharge simulations vs. observations.

            Arguments
            ---------
            freq : <pandas frequency>
                Any pandas frequency to aggregate to. Can only aggregate to a
                frequency lower than in the data! Default: as given in file.
            plotkw:
                Keywords to swimpy.plot.plot().
            """
            freq = freq or self._time[0]
            df_obssim = pd.concat(self.obs_sim_overlap(), keys=['obs', 'sim'])
            if not df_obssim.empty:
                df = df_obssim.unstack(level=0)
                df.columns.names = ['station', 'variable']
                df.columns=df.columns.swaplevel(0,1)
                plot.plot(df, 'station', freq=freq, **plotkw)
            return

        if (self._exists and 'discharge' in vars and
            hasattr(self.project, 'stations') and
            hasattr(self.project.stations, self._time+'_discharge_observed')):
            
            obs = getattr(self.project.stations, self._time+'_discharge_observed')
            if(any(s in obs.columns for s in stats)):
                setattr(self, 'obs_sim_overlap', obs_sim_overlap)
                # methods as attributes without pandas columns warning
                for a in ['NSE', 'rNSE', 'pbias', 'pbias_abs',
                        'plot_discharge_comparison']:
                    self._set_attr(a, locals()[a]())
        
        def plot_flow_duration(stations=None, ax=None, **linekw):
            stations = stations or self.columns.get_level_values(self._space).unique()
            stations = stations if type(stations) is list else \
                        [stations] if type(stations) in [int, str] else \
                        list(stations)
            lines = []
            for s in stations:
                fd = hydro.flow_duration(self['discharge'][s])
                line = plot.plot_flow_duration(fd, ax=ax, **linekw)
                lines.append(line)
            return lines
        
        def plot_flow_duration_polar(station, percentilestep=10, freq='m',
                                    colormap='jet_r', ax=None, **barkw):
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
            ax = plot.plot_flow_duration_polar(self['discharge'][station], freq=freq, ax=ax,
                                            percentilestep=percentilestep,
                                            colormap=colormap, **barkw)
            return ax
        
        if self._exists and 'discharge' in vars:
            setattr(self, 'plot_flow_duration', plot_flow_duration)
            setattr(self, 'plot_flow_duration_polar', plot_flow_duration_polar)

        # TODO: strange values of river_runoff
        # # runoff and precipitation
        # def runoff_coefficient():
        #     (_, p), (_, r) = self[['precipitation', 'river_runoff']].items()
        #     return r/p
        
        # if 'precipitation' in vars and 'river_runoff' in vars:
        #     setattr(self, 'runoff_coefficient', runoff_coefficient)

        return

    @property
    def path(self):
        relpath = osp.join(self.project.outputpath, self.file)
        return self._path if hasattr(self, '_path') else relpath

    @path.setter
    def path(self, value):
        self._path = value
        return
    
    @property
    def _exists(self):
        return osp.exists(self.path)
    
    @property
    def _space(self):
        fsplt = self.file.split('_')
        return '_'.join(fsplt[0:2]) if fsplt[1] == 'label' else fsplt[0]

    @property
    def _time(self):
        fsplt = self.file.split('_')
        return fsplt[2] if fsplt[1] == 'label' else fsplt[1]

    @property
    def _name(self):
        fsplt = self.file.split('_')
        nameext = fsplt[3] if fsplt[1] == 'label' else fsplt[2]
        return nameext.removesuffix('.csv')
    
    def _set_attr(self, name, attr):
        """methods as attributes without pandas columns warning"""
        setattr(self, name, '')
        setattr(self, name, attr)
        return

    def from_csv(self, path=None, **kwargs):
        """Read output file. Result object inherits from pandas.DataFrame. 

        Arguments
        ---------
        path: str, optional
            File other than default to read.
        **kwargs :
            Keywords to pandas.read_csv.
        """
        if path == self.path:
            path = None
        if self._exists or path:
            path = path or self.path
            df = utils.read_csv_multicol(path, 'time', self._time[0],
                                self._space, **kwargs)
        else:
            df = None
        return df
    
    from_project = from_csv

    def write(self, path=None, **kwargs):
        """Write to csv file.

        Arguments
        ---------
        path: str, optional
            File other than default to write to.
        **kwargs :
            Keywords to pandas.to_csv.
        """
        if path == self.path:
            path = None
        if self._exists or path:
            path = path or self.path
            utils.write_csv_multicol(self, path, self._space, **kwargs)
        return
    
    to_csv = write

    def peak_over_threshold(self, percentile=1, threshold=None, maxgap=None,
                            stations=None, variables=None):
        """Identify peaks over a threshold, return peak value, length, date and recurrence.

        Arguments
        ---------
        percentile : number
            The percentile threshold of values, e.g. 1 means Q1 (probability of exceedance).
        threshold : number, optional
            Absolute threshold to use for peak identification.
        maxgap : int, optional
            Largest gap between two threshold exceedance periods to count as
            single peak event. Number of timesteps. If not given, every
            exceedance is counted as individual peak event.
        stations : stationID | list of stationIDs
            Return subset of stations. Default all.
        variables : str | list of variables
            Variables to which this function is to be applied. Default all.

        Returns
        -------
        pd.DataFrame :
            Peak values ordered dataframe with order index and peak,
            length, peak date and recurrence columns with MultiIndex if more
            than one station and variable is selected.
        """
        stations = stations or self.columns.get_level_values(self._space).unique()
        stations = stations if type(stations) is list else \
                    [stations] if type(stations) in [int, str] else \
                    list(stations)
        variables = variables or self.columns.get_level_values('variable').unique()
        variables = variables if type(variables) is list else \
                    [variables] if type(variables) in [int, str] else \
                    list(variables)
        kw = dict(percentile=percentile, threshold=threshold, maxgap=maxgap)
        pot = [hydro.peak_over_threshold(self[v][s], **kw) for s in stations for v in variables]
        out = pot[0] if len(stations) == 1 and len(variables) == 1 else \
                pd.concat(pot, keys=itertools.product(stations, variables))
        out.index.names = ['order'] if len(out.index.names) == 1 else [self._space, 'variable', 'order']
        return out
    
    # TODO: revise! Works but maybe not as it should (returns no ax object etc.)
    def plot(self, freq=None, **plotkw):
        """Line plot of selected data with preprocessing.

        Arguments
        ---------
        freq : <pandas frequency>
            Any pandas frequency to aggregate to. Can only aggregate to a
            frequency lower than in the data! Default: as given in file.
        plotkw:
            Keywords to swimpy.plot.plot().
        """
        if not self._exists:
            raise IOError('File {} does not yet exist. You need to run SWIM first.'.format(osp.relpath(self.path, self.project.projectdir)))
        freq = freq or self._time[0]
        plot.plot(self, self._space, freq=freq, **plotkw)
        return
    
    def to_grass(self, variable=None, timestep=None, prefix=None,
                 name=None, strds=True, mapset=None):
        space = self._space.split('_')[0]
        reclasser = getattr(getattr(self.project, space), 'reclass')
        _to_raster(
            self.project, self, reclasser, variable = variable, timestep=timestep,
            prefix=prefix, name=name, strds=strds, mapset=mapset)
        return
    to_grass.__doc__ = _to_raster.__doc__

    def __repr__(self):
        if self._exists:
            rpr = super().__repr__()
        else:
            rpr = '<File {} does not yet exist. You need to run SWIM first.>'.format(osp.relpath(self.path, self.project.projectdir))
        return rpr


class output_files(dict):
    """
    Set or get values from output.nml.

    Get: output_files(<filename>)

    Set: output_files(<filename>=[var1, var2])
         where <filename> has to follow SWIM <space>_<time>_<name> convention
         for output files and [var1, var2] must be valid variable names. File
         ending is determined automatically and must not be given.
    """

    def __init__(self, project):
        self.project = project
        self.read()
        return
    
    def _create_propertyplugins(self):
        """
        Output files as project attributes
        """
        plugins = {}
        for k in self.keys():
            kfle = k + '.' + self.project.output_parameters['output_default_format']
            class _of(OutputFile):
                file = kfle
                project = self.project
            _of.__name__ = k
            plugins[k] = propertyplugin(_of)
        return plugins

    @property
    def path(self):
        relpath = osp.join(self.project.inputpath, 'output.nml')
        return self._path if hasattr(self, '_path') else relpath

    @path.setter
    def path(self, value):
        self._path = value
        return
    
    def __call__(self, *get, **set):
        assert get or set
        if set:
            self.update(set)
            self.write()
            self.interfaces = self._create_propertyplugins()
            self.project.settings(**self.interfaces)
        if get:
            return [self[k] for k in get]
        return
    
    def read(self, path=None):
        """
        Read output file definitions from project.output_files.path or a
        different .nml file.

        Arguments
        ---------
        path: str, optional
            A file name from which output definitions are read. By default
            project.output_files.path is used. If given, path will be updated.
        """
        path = path or self.path
        self._path = path
        nml = f90nml.read(path)
        nml = {'_'.join([v.get('space_name', v['space']), v['time'], v['name']]): v['variables']
               for v in nml.values()}
        # make sure self is fully (re-)initialised
        self.clear()
        super().__init__(nml)
        # output files as project attributes
        self.interfaces = self._create_propertyplugins()
        self.project.settings(**self.interfaces)
        return 
    
    def write(self, path=None):
        """
        Write output file definitions into project.output_files.path or a new .nml file.

        Arguments
        ---------
        path: str, optional
            A file name into which output definitions are written. By default
            project.output_files.path is used. If given, path will NOT be
            updated.
        """
        path = path or self.path
        # order of ospace is important for checks below!
        ospace = ('hydrotope_label', 'hydrotope',
                  'subbasin_label', 'subbasin', 'catchment')
        otime = ('daily', 'monthly', 'annual')
        nml = []
        for k, v in self.items():
            ksplt = k.split('_')
            space = '_'.join(ksplt[0:2]) if ksplt[1] == 'label' else ksplt[0]
            if space not in ospace:
                raise KeyError("Invalid file name '{}'; unsupported space attribute (first element)!".format(k))
            time = ksplt[2] if ksplt[1] == 'label' else ksplt[1]
            if time not in otime:
                raise KeyError("Invalid file name '{}'; unsupported time attribute (second element)!".format(k))
            name = '_'.join(ksplt[3:]) if ksplt[1] == 'label' else '_'.join(ksplt[2:])
            knml = f90nml.Namelist({'file': {'name': name, 'space': space, 'time': time, 'variables': v}})
            nml.append(knml)
        with open(path, "w") as f:
            for fe in nml:
                fe.write(f)
        return
    
    def __repr__(self) -> str:
        rpr = '<{}: {}>\n'.format(self.__class__.__name__,
                                  osp.relpath(self.path, self.project.projectdir))
        odir = "Files with variables in directory '{}':\n".format(self.project.output_parameters['output_dir'])
        files = 'dict {\n'
        for f, v in self.items():
            vars = ', '.join(v) if isinstance(v, list) else v
            files += f + ': [' + vars + ']\n'
        return rpr + odir + files + '}'


# classes attached to project in defaultsettings
PLUGINS = {n: globals()[n] for n in ['output_files']}
