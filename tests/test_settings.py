"""These are test settings of SWIMpy."""
import os.path as osp
from modelmanager.plugins.grass import GrassAttributeTable as _GAT
import pandas as pd

# path outside project dir dynamic relative to resourcedir to work with clones
grass_db = property(lambda p: osp.join(osp.realpath(p.resourcedir), '..', '..', 'grassdb'))
grass_location = "utm32n"
grass_mapset =  "swim"
grass_setup = dict(elevation = "elevation@PERMANENT",
                   stations = "stations@PERMANENT",
                   upthresh=40, lothresh=11,
                   landuse = "landuse@PERMANENT",
                   soil = "soil@PERMANENT")

class stations(_GAT):
    vector = 'stations_snapped@swim'
    key = 'NAME'
    _cols = ['y', 'm', 'd', 'BLANKENSTEIN']
    _path = osp.join(osp.dirname(__file__), '../input/runoff.dat')
    daily_discharge_observed = pd.read_table(
        _path, skiprows=2, header=None, delim_whitespace=True,
        index_col=0, parse_dates=[[0, 1, 2]], names=_cols, na_values=[-9999])
    daily_discharge_observed['HOF'] = daily_discharge_observed['BLANKENSTEIN']*0.5
