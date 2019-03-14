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


def _read_q():
    cols = ['y', 'm', 'd', 'BLANKENSTEIN']
    path = osp.join(osp.dirname(__file__), '../input/runoff.dat')
    q = pd.read_csv(path, skiprows=2, header=None, delim_whitespace=True,
                    index_col=0, parse_dates=[[0, 1, 2]], names=cols,
                    na_values=[-9999])
    q.index = q.index.to_period()
    q['HOF'] = q['BLANKENSTEIN']*0.5
    return q


class stations(_GAT):
    vector = 'stations_snapped@swim'
    key = 'NAME'
    daily_discharge_observed = _read_q()
