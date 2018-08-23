"""These are test settings of SWIMpy."""
from modelmanager.plugins.grass import GrassAttributeTable as _GAT

grass_db = "../grassdb"
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
    return ro.to_dict()


class stations(_GAT):
    vector = 'stations_snapped'
    key = 'NAME'
    add_attributes = {'daily_discharge_observed': 'get_station_daily_qobs'}
