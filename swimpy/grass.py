"""A module to interface SWIM and GRASS.

The main classes/plugins are interfaces to the *m.swim* GRASS modules and can
execute them without ever starting GRASS if the following projects are given:

grass_db : str path
    The path to the GRASS database directory.
grass_location : str
    The location name.
grass_mapset : str
    The mapset name to execute the module in, i.e. new maps are created here.

The modules can either be execute as is by:

>>> subbasins.create()

Or with the `GRASS_OVERWRITE` environment variable set and followed by the
``postprocess`` method like this:

>>> subbasins.update()

Input arguments to the module can either be given as project settings, as
class/plugin attributes or when calling the ``create`` or ``update`` methods.
"""
from __future__ import absolute_import
import os
import os.path as osp

import numpy as np
import pandas as pd
from modelmanager.utils import propertyplugin
from modelmanager.plugins import grass as mmgrass

# enable from swimpy.grass import GrassAttributeTable for convenience
GrassAttributeTable = mmgrass.GrassAttributeTable


def reclass_raster(project, inrast, outrast, values, mapset=None,
        float_precision=6):
    """Reclass inrast with int/float 'values' to outrast.

    Arguments
    ---------
    project : <swimpy.Project>
        A project instance with GRASS setttings.
    inrast : str
        Name of input raster to be reclassed.
    outrast : str
        Name of raster to be created.
    values : list-like | <pandas.Series>
        Values the raster is reclassed to. If a pandas.Series is parsed, the
        index is used to create the mapping, otherwise the categories are
        assumed to be ``range(1, len(values)+1)``.
    mapset : str, optional
        Different than the default `grass_mapset` mapset to write to.
    precision : int
        Decimal places converted to raster if values are floats.
    """
    from pandas import Series
    if isinstance(values, Series):
        ix = values.index
    else:
        ix = range(1, len(values)+1)
    # Convert to ints to work with r.reclass
    if not (hasattr(values, 'dtype') and values.dtype == int):
        values = (values*10**float_precision).astype(int)
        outrast += '__int'
    columns = np.column_stack([ix, values])
    mapset = mapset or project.grass_mapset
    with mmgrass.GrassSession(project, mapset=mapset) as grass:
        tf = grass.tempfile()
        np.savetxt(tf, columns, delimiter='=', fmt="%i")
        grass.run_command('r.reclass', input=inrast, output=outrast, rules=tf)
        if outrast.endswith('__int'):
            grass.mapcalc("'%s'=float('%s')/%i" % (
                outrast[:-5], outrast, 10**float_precision))
            grass.run_command('g.remove', type='raster', name=outrast,
                flags='f')
    os.remove(tf)
    return 0


# generic `to_raster` function to be used as method in output.py
# values are the ProjectOrRunData instance and reclasser the subbasins or
# hydroptes.reclass_raster methods (not in docstring as it's copied)
def _subbasin_or_hydrotope_values_to_raster(
        project, values, reclasser,
        timestep=None, prefix=None, name=None, strds=True, mapset=None):
    """Create GRASS raster from values for each timestep.

    Arguments
    ---------
    values : pd.DataFrame | pd.Series
        Values to reclass raster to with index of time intervals and columns
        subbasins or hydrotopes.
    timestep : str | list or str, optional
        Select individual timestep (str) or several (list). Default is
        all timesteps.
    prefix : str, optional
        Different prefix. Default is name of class (`hydrotope_*`). The
        timestep is appended at the end.
    name : str | list, optional
        To set a name without timestep at the end.
    strds : bool
        Create a space-time raster dataset if len(timestep)>1 named after
        `prefix` (trailing _ trimmed).
    mapset : str, optional
        Mapset to write to. Defaults to `grass_mapset`.
    """
    values = values.to_frame().T if isinstance(values, pd.Series) else values
    # argument preparation
    prefix = prefix or values.__class__.__name__
    if timestep:
        assert type(timestep) in [str, list, slice]
        d = values.loc[[timestep] if type(timestep) == str else timestep]
    else:
        d = values.copy()
    if name:
        assert type(name) in [str, list]
        names = [name] if type(name) == str else name
        assert len(names) == len(d), 'Wrong number of names parsed.'
    else:
        names = [prefix+'_'+str(i) for i in d.index]
    mapsetlab = '@'+mapset if mapset else ''
    # do the reclassing
    for i, n in enumerate(names):
        reclasser(d.iloc[i, :], n, mapset=mapset)
        if int(os.environ.get('GRASS_VERBOSE', 1)):
            print('Created raster %s' % (n+mapsetlab))
    # create spacetime ds
    if strds and len(d.index) > 1 and hasattr(d.index, 'freq'):
        dsname = prefix.strip('_')
        with mmgrass.GrassSession(project, mapset=mapset) as grass:
            for n, ts in zip(names, d.index.to_timestamp().date):
                date = ts.strftime('%d %b %Y')
                grass.run_command('r.timestamp', map=n, date=date)
            grass.run_command('t.create', output=dsname, title=dsname,
                              description='created with swimpy')
            grass.run_command('t.register', input=dsname, maps=','.join(names))
        if int(os.environ.get('GRASS_VERBOSE', 1)):
            print('Created space-time raster dataset %s' % (dsname+mapsetlab))
    return
