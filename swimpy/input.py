"""
SWIM input functionality.
"""
import pandas as pd
from modelmanager.plugins.templates import TemplatesDict as _TemplatesDict

from swimpy import utils


class basin_parameters(_TemplatesDict):
    """
    Set or get any values from the .bsn file by variable name.
    """
    template_patterns = ['input/*.bsn']


class config_parameters(_TemplatesDict):
    """
    Set or get any values from the .cod or swim.conf file by variable name.
    """
    template_patterns = ['input/*.cod', 'swim.conf']


class subcatch_parameters(utils.ReadWriteDataFrame):
    """
    Read or write parameters in the subcatch.prm file.
    """
    path = 'input/subcatch.prm'

    def read(self, **kwargs):
        bsn = pd.read_table(self.path, delim_whitespace=True)
        stn = 'stationID' if 'stationID' in bsn.columns else 'station'
        bsn.set_index(stn, inplace=True)
        pd.DataFrame.__init__(self, bsn)
        return

    def write(self, **kwargs):
        bsn = self.copy()
        bsn['stationID'] = bsn.index
        strtbl = bsn.to_string(index=False, index_names=False)
        with open(self.path, 'w') as f:
            f.write(strtbl)
        return