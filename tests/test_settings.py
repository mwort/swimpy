"""These are test settings of SWIMpy."""
import os.path as osp
from modelmanager.plugins.grass import GrassAttributeTable as _GAT
import pandas as pd

# path outside project dir dynamic relative to resourcedir to work with clones
grass_db = property(lambda p: osp.join(osp.realpath(p.resourcedir), '..', '..', 'grassdb'))
grass_location = "utm32n"
grass_mapset = "swim"
grass_setup = dict(subbasin_id = "subbasins", subbasins = "subbasins",
                   elevation="elevation@PERMANENT",
                   stations="stations@PERMANENT",
                   upthresh=50, lothresh=1.6, streamthresh=200,
                   minmainstreams=50, contours=200,
                   predefined="reservoirs@PERMANENT",
                   landuse_id="landuse@PERMANENT",
                   soil_id="soil@PERMANENT")

cluster_slurmargs = dict(qos='priority')


# import of discharge observations
def _read_q():
    path = osp.join(osp.dirname(__file__), '../input/discharge.csv')
    q = pd.read_csv(path, index_col=0, parse_dates=[0], na_values=[-9999])
    q.index = q.index.to_period()
    q['HOF'] = q['BLANKENSTEIN']*0.5
    return q


# discharge station data
class stations(_GAT):
    # GRASS vector file of station locations
    vector = 'stations_snapped@swim'
    # key column in station vector file; used as 'station_id' in catchment.csv
    # which is linked to column names in discharge.csv
    key = 'NAME'
    # corresponding discharge observations as pd.DataFrame
    daily_discharge_observed = _read_q()
