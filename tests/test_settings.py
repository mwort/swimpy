"""These are test settings of SWIMpy."""
import os.path as osp
from modelmanager.plugins.grass import GrassAttributeTable as _GAT

# path outside project dir dynamic relative to resourcedir to work with clones
grass_db = property(lambda p: osp.join(osp.realpath(p.resourcedir), '..', '..', 'grassdb'))
grass_location = "utm32n"
grass_mapset =  "swim"
grass_setup = dict(elevation = "elevation@PERMANENT",
                   stations = "stations@PERMANENT",
                   upthresh=40, lothresh=11,
                   landuse = "landuse@PERMANENT",
                   soil = "soil@PERMANENT")

def get_station_daily_qobs(project):
    ro = project.station_daily_discharge_observed
    ro['HOF'] = ro['BLANKENSTEIN']*0.5
    return ro.to_dict('series')


class stations(_GAT):
    vector = 'stations_snapped@swim'
    key = 'NAME'
    add_attributes = {'daily_discharge_observed': 'get_station_daily_qobs'}
