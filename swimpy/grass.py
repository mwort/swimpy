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
import math

import numpy as np
import pandas as pd
from modelmanager.plugins import grass as mmgrass

# enable from swimpy.grass import GrassAttributeTable for convenience
GrassAttributeTable = mmgrass.GrassAttributeTable


def get_modulearg(project, module, arg):
    """Get value of GRASS m.swim.* module argument.
    Either derived from 'grass_setup' in settings.py
    or the module's default value is returned.
    
    Arguments
    ---------
    project : <swimpy.Project>
        A project instance with GRASS setttings.
    module : str
        Name of m.swim.* module.
    arg : str
        Name of <module> argument for which the value shall be retrieved.
    """
    if arg in getattr(project, 'grass_setup'):
        out = project.grass_setup[arg]
    else:
        with mmgrass.GrassSession(project):
            from grass.pygrass.modules import Module
            mod = Module(module, run_=False)
            plist = {p.name: p.value for p in mod.params_list}
            if arg not in plist:
                raise KeyError("{} is not a parameter of {}!".format(arg, module))
            out = plist[arg]
            if out is None and arg in mod.required:
                raise KeyError("{} is a required parameter and must be given in settings.py!".format(arg))
    return out            


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
    float_precision : int
        Decimal places converted to integer if values are floats. Precision is
        automatically reduced if 32 bit range is exceeded.
    """
    from pandas import Series
    if isinstance(values, Series):
        ix = values.index
    else:
        ix = range(1, len(values)+1)
    # Convert to ints to work with r.reclass
    if not (hasattr(values, 'dtype') and values.dtype == int):
        valmax = max(abs(values))*10**float_precision
        vallim = 2**32/2 - 1
        if valmax > vallim:
            mag = math.ceil(math.log(valmax, 10))
            float_precision -= mag - 9
            if float_precision < 0:
                raise ValueError('The dataset contains very large numbers that '
                                 'cannot be represented by GRASS!')
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
# values are the ReadWriteDataFrame instance and reclasser the subbasin,
# hydropte, or catchment.reclass methods (not in docstring as it's copied)
def _to_raster(
        project, values, reclasser,
        timestep=None, variable=None, prefix=None, name=None, strds=True,
        mapset=None):
    """Create GRASS raster from values for each timestep.

    Arguments
    ---------
    variable : str | list, optional
        Select individual (str) or multiple (list) variables. Default is
        all variables.
    timestep : str | list or str, optional
        Select individual timestep (str) or several (list). Default is
        all timesteps.
    prefix : str, optional
        GRASS raster name prefix. Default is name of class of values. Timestep
        and variable are appended at the end.
    name : str | list, optional
        Full GRASS raster name. One name for each timestep x variable
        combination.
    strds : bool
        For each variable, create a space-time raster dataset if
        len(timestep)>1 named after `prefix` (trailing _ trimmed).
    mapset : str, optional
        Mapset to write to. Defaults to `grass_mapset`.
    """
    values = values.to_frame().T if isinstance(values, pd.Series) else values
    variable = variable or list(values.columns.get_level_values('variable').unique())
    variable = [variable] if type(variable) == str else variable
    # argument preparation
    prefix = prefix or values.__class__.__name__
    if timestep:
        assert type(timestep) in [str, list, slice]
        d = values.loc[[timestep] if type(timestep) == str else timestep]
    else:
        d = values.copy()
    d = d[variable]
    if name:
        assert type(name) in [str, list]
        names = [name] if type(name) == str else name
        assert len(names) == (len(d)*len(d.columns.get_level_values('variable').unique())), 'Wrong number of names parsed.'
    mapsetlab = '@'+mapset if mapset else ''
    # do the reclassing
    i = 0
    nd = {}
    for v in variable:
        nn = []
        for t in d.index:
            if name:
                n = names[i]
            else:
                n = prefix+'_'+str(t)+'_'+v
            nn.append(n)
            reclasser(d.loc[t][v], n, mapset=mapset)
            if int(os.environ.get('GRASS_VERBOSE', 1)):
                print('Created raster %s' % (n+mapsetlab))
            i += 1
        nd.update({v: nn})
    # create spacetime ds
    if strds and len(d.index) > 1 and hasattr(d.index, 'freq'):
        dsname = prefix.strip('_')
        with mmgrass.GrassSession(project, mapset=mapset) as grass:
            for v in variable:
                for n, ts in zip(nd[v], d.index.to_timestamp().date):
                    date = ts.strftime('%d %b %Y')
                    grass.run_command('r.timestamp', map=n, date=date)
                tname = dsname+'_'+v
                grass.run_command('t.create', output=tname, title=tname,
                                description='created with swimpy')
                grass.run_command('t.register', input=tname, maps=','.join(nd[v]))
                if int(os.environ.get('GRASS_VERBOSE', 1)):
                    print('Created space-time raster dataset %s' % (tname+mapsetlab))
    return
