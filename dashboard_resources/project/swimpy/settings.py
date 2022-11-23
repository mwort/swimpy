"""These are test settings of SWIMpy."""
import os.path as osp
import os
import datetime as dt
from modelmanager.plugins.grass import GrassAttributeTable as _GAT
import numpy as np
import pandas as pd
from modelmanager.plugins.pandas import ProjectOrRunData as _ProjectOrRunData
from modelmanager.plugins.templates import TemplatesDict as _TempDict
from modelmanager.utils import propertyplugin as _propertyplugin

from swimpy.grass import hydrotopes as _swimpy_hydrotopes
from swimpy import input, output
from swimpy.dashboard import App as dashboard


# path outside project dir dynamic relative to resourcedir to work with clones
grass_db = property(lambda p: osp.join(osp.realpath(p.resourcedir), '..', '..', 'grassdb'))
grass_location = "utm32n"
grass_mapset = "swim"
grass_setup = dict(elevation="elevation@PERMANENT",
                   stations="stations@PERMANENT",
                   upthresh=50, lothresh=1.6, streamthresh=200,
                   predefined="reservoirs@PERMANENT",
                   landuse="landuse@PERMANENT",
                   soil="soil@PERMANENT")

cluster_slurmargs = dict(qos='priority')

save_run_parameters = False
save_run_files = [
    "station_daily_discharge",
    "hydrotope_daily_waterbalance",
    "reservoir_output",
    "catchment_annual_waterbalance",
    "catchment_daily_waterbalance",
    "catchment_daily_temperature_precipitation",
    "hydrotope_evapmean_gis",
    "hydrotope_gwrmean_gis",
    "hydrotope_pcpmean_gis", 
]

class climate(input.climate):
    variable_names = ["tas", "tasmin", "tasmax", "pr", "rsds", "hurs"]
    options = {
        "Observations (EOBSv25)": ("EOBSv25_MDK_grid_ncinfo.nml", (1951, 2021)),
        "RCP 4.5 (ICHEC-EC-EARTH)": ("MDK_ICHEC-EC-EARTH_rcp45_r12i1p1_KNMI-RACMO22E_v1_ncinfo.nml", (1951, 2098)),
        "RCP 2.6 (CCCMA-CANESM2)": ("MDK_CCCma-CanESM2_rcp26_r4i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 4.5 (MOHC-HADGEM2)": ("MDK_MOHC-HadGEM2-ES_rcp45_r1i1p1_KNMI-RACMO22E_v2_ncinfo.nml", (1951, 2098)),
        "RCP 8.5 (MPI-M-MPI-ESM)": ("MDK_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 8.5 (MOHC-HADGEM2)": ("MDK_MOHC-HadGEM2-ES_rcp85_r1i1p1_KNMI-RACMO22E_v2_ncinfo.nml", (1951, 2098)),
        "RCP 2.6 (MPI-M-MPI-ESM)": ("MDK_MPI-M-MPI-ESM-LR_rcp26_r3i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 8.5 (MIROC-MIROC5)": ("MDK_MIROC-MIROC5_rcp85_r1i1p1_CLMcom-CCLM4-8-17_v1_ncinfo.nml", (1951, 2100)),
        "RCP 2.6 (ICHEC-EC-EARTH)": ("MDK_ICHEC-EC-EARTH_rcp26_r12i1p1_CLMcom-CCLM4-8-17_v1_ncinfo.nml", (1951, 2100)),
        "RCP 4.5 (NCC-NORESM1-M)": ("MDK_NCC-NorESM1-M_rcp45_r1i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 2.6 (CCCMA-CANESM2)": ("MDK_CCCma-CanESM2_rcp26_r2i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 2.6 (MIROC-MIROC5)": ("MDK_MIROC-MIROC5_rcp26_r1i1p1_CLMcom-CCLM4-8-17_v1_ncinfo.nml", (1951, 2100)),
        "RCP 4.5 (ICHEC-EC-EARTH)": ("MDK_ICHEC-EC-EARTH_rcp45_r12i1p1_CLMcom-CCLM4-8-17_v1_ncinfo.nml", (1951, 2100)),
        "RCP 4.5 (CCCMA-CANESM2)": ("MDK_CCCma-CanESM2_rcp45_r5i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 2.6 (MOHC-HADGEM2)": ("MDK_MOHC-HadGEM2-ES_rcp26_r1i1p1_KNMI-RACMO22E_v2_ncinfo.nml", (1951, 2098)),
        "RCP 4.5 (ICHEC-EC-EARTH)": ("MDK_ICHEC-EC-EARTH_rcp45_r2i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 8.5 (ICHEC-EC-EARTH)": ("MDK_ICHEC-EC-EARTH_rcp85_r2i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 8.5 (CCCMA-CANESM2)": ("MDK_CCCma-CanESM2_rcp85_r4i1p1_DWD-EPISODES2018_v1-r1_ncinfo.nml", (1951, 2100)),
        "RCP 4.5 (MPI-M-MPI-ESM)": ("MDK_MPI-M-MPI-ESM-LR_rcp45_r1i1p1_CLMcom-CCLM4-8-17_v1_ncinfo.nml", (1951, 2100)),
    }
    def set_input_climfiles(self, option):
        oclims, (sty, eny) = self.options[option]
        climdir = self.project.config_parameters['climatedir']
        climfls = [osp.join(climdir, "clim%i.dat" % i)
                   for i in (1, 2)]
        # check before overwriting
        for path in climfls:
            assert osp.islink(path) or not osp.exists(path), f"{path} exists and is not a link."
        for path, new in zip(climfls, oclims):
            if osp.islink(path):
                os.remove(path)
            target = osp.realpath(osp.join(climdir, new))
            os.symlink(target, path)
        # change start date
        self.project.config_parameters(iyr=sty)
        myrs = eny - sty + +1
        if self.project.config_parameters["nbyr"] > myrs:
            self.project.config_parameters(nbyr=myrs)
        return

    def set_input(self, option):
        oclim, (sty, eny) = self.options[option]
        opath = osp.join(self.project.config_parameters['climatedir'], "ncinfo.nml")
        target = osp.join(self.project.config_parameters['climatedir'], oclim)
        # check before overwriting
        assert osp.islink(opath) or not osp.exists(opath), f"{opath} exists and is not a link."
        if osp.islink(opath):
            os.remove(opath)
        os.symlink(osp.relpath(target, osp.dirname(opath)), opath)

        # change start and run time
        iyr = self.project.config_parameters["iyr"]
        if iyr < sty or iyr > eny:
            self.project.config_parameters(iyr=sty)
            iyr=sty
        myrs = eny - iyr + 1
        if self.project.config_parameters["nbyr"] > myrs:
            self.project.config_parameters(nbyr=myrs)
        return


@_propertyplugin
class catchment_daily_temperature_precipitation(_ProjectOrRunData):

    path = "none"

    def from_project(self, path, **kw):
        stend = (self.project.config_parameters.start_date,
                 self.project.config_parameters.end_date)
        key = ["tmean", "precipitation"]
        tp = pd.concat([self.project.climate.netcdf_inputdata.read(k, time=stend)
                        for k in key], axis=1, keys=key)
        # not weighting by subbasin size
        return tp.mean(level=0, axis=1)

    @staticmethod
    def from_csv(path, **readkwargs):
        df = pd.read_csv(path, index_col=0, parse_dates=[0], **readkwargs)
        df.index = df.index.to_period(freq='d')
        return df


def _read_q():
    cols = ['y', 'm', 'd', 'BLANKENSTEIN']
    path = osp.join(osp.dirname(__file__), '../input/runoff.dat')
    q = pd.read_csv(path, skiprows=2, header=None, delim_whitespace=True,
                    index_col=0, parse_dates=[[0, 1, 2]], names=cols,
                    na_values=[-9999])
    q.index = q.index.to_period()
    q['HOF'] = q['BLANKENSTEIN']*0.5
    return q


class stations:
    def __init__(self, project):
        self.project = project

    @property
    def daily_discharge_observed(self):
        return _read_q()


@_propertyplugin
class station_daily_discharge(output.station_daily_discharge):
    @staticmethod
    def from_project(path, **readkwargs):
        df = output.station_daily_discharge.from_project(path, **readkwargs)
        return df.drop("observed", axis=1)


class hydrotopes(_swimpy_hydrotopes):

    array_file = osp.join(osp.dirname(__file__), "hydrotopes.bin")
    array_na = 0
    array_shape = (446, 482)
    array_latlon_bounds = [(50.50394243, 11.56237908), (50.08653291, 12.21395716)]

    @property
    def array(self):
        array = np.fromfile(self.array_file, dtype=np.int32).reshape(self.array_shape)
        return array

    def to_image(self, hydrotope_series, output_path, vminmax, cmap=None):
        from matplotlib.pyplot import imsave

        # reclass hyd array
        fltarr = self.array.flatten()
        newarr = np.zeros_like(fltarr, dtype=float)
        newarr[fltarr != self.array_na] = hydrotope_series[fltarr[fltarr != self.array_na]]
        newarr[fltarr == self.array_na] = np.nan
        recarr = np.reshape(newarr, self.array.shape)
        imsave(output_path, recarr, origin="upper", vmin=vminmax[0], vmax=vminmax[1], cmap=cmap)
        return


def reservoir_output(project):
    path = osp.join(project.projectdir, "output", "Res", "reservoir_FoermitzTS.out")
    df = pd.read_csv(path, delim_whitespace=True)
    dtms = [dt.date(y, 1, 1) + dt.timedelta(d - 1)
                for y, d in zip(df.pop('YEAR'), df.pop('DAY'))]
    df.index = pd.PeriodIndex(dtms, freq='d', name='time')
    return df

@_propertyplugin
class reservoir_parameters(_TempDict):
    template_patterns = ['input/reservoir.ctrl']

