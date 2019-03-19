import os.path as osp
from modelmanager.plugins.grass import GrassAttributeTable as _GAT
import pandas as pd

# path to SWIMpy resource directory to use in settings only
_here = osp.dirname(__file__)

# cloning
clone_ignore = ['output/*/*']
clone_links = ['input/Sub', 'input/Soils']

# GRASS
grass_db = osp.join(_here, '../../grassdb')
grass_location = "utm32n"
grass_mapset = "swim"
grass_setup = {'elevation': "elevation@PERMANENT",  # any m.swim.* argument
               'stations': "stations@PERMANENT",
               'upthresh': 40, 'lothresh': 11,
               'landuse': "landuse@PERMANENT",
               'soil': "soil@PERMANENT"}

# cluster
cluster_slurmargs = {'qos': 'priority',
                     'time': 10}


# stations and observed discharge
class stations(_GAT):
    vector = 'stations_snapped@swim'
    key = 'NAME'
    daily_discharge_observed = pd.read_csv(
        osp.join(_here, 'daily_discharge_observed.csv'),
        index_col=0, parse_dates=[0], date_parser=pd.Period)


# run save
run_save_files = ['station_daily_discharge']
run_save_indicators = ['station_daily_discharge.NSE',
                       'station_daily_discharge.pbias']
