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

from modelmanager.utils import propertyplugin
from modelmanager.plugins import grass as mmgrass


class Subbasins(mmgrass.GrassModulePlugin):
    """Plugin to deal with subbasin-related input creation.

    The following attributes are either needed as project settings,
    class/instance attributes or may be parsed to the create and update method:

    Attributes
    ----------
    elevation : str
        Elevation raster name.
    stations : str
        Point vector of gauging stations.
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    <any other m.swim.subbasins argument>, optional
        Any other argument for m.swim.subbasins will be parsed.

    Note
    ----
    The `subbasins` argument must be given as attribute as giving it as project
    setting would overwrite the plugin which has the same name.
    """
    module = 'm.swim.subbasins'
    #: Subbasins vector (w/o mapset) to be created in `grass_mapset`
    vector = 'subbasins'
    #: Subbasins raster (w/o mapset) to be created in `grass_mapset`
    raster = 'subbasins'
    subbasins = property(lambda self: self.vector)

    def postprocess(self, **moduleargs):
        self.project.config_parameters['mb'] = len(self.attributes)
        self.project.routing(**moduleargs)
        self.project.substats(**moduleargs)
        self.project.subcatch_definition.update()
        self.project.hydrotopes(**moduleargs)
        # TODO: write nc_climate files
        return

    @propertyplugin
    class attributes(mmgrass.GrassAttributeTable):
        """The subbasins attribute table as a ``pandas.DataFrame`` object."""
        vector = property(lambda self: self.project.subbasins.vector)

    def reclass(self, values, outrast, mapset=None):
        """Reclass subbasin raster to 'values'.

        Arguments
        ---------
        outrast : str
            Name of raster to be created.
        values : list-like | <pandas.Series>
            Values the raster is reclassed to. If a pandas.Series is parsed,
            the index is used to create the mapping, otherwise the categories
            are assumed to be ``range(1, len(values)+1)``.
        mapset : str, optional
            Different than the default `grass_mapset` mapset to write to.
        """
        sbname = self.raster + '@' + self.project.grass_mapset
        reclass_raster(self.project, sbname, outrast, values, mapset=mapset)
        return


class Routing(mmgrass.GrassModulePlugin):
    """Plugin to deal with routing-related input.

    Subbasins vector and accumulation raster must exist.

    The following attributes are either needed as project settings,
    class/instance attributes or may be parsed to the create and update method:

    Attributes
    ----------
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    figpath : str, optional
        Path to fig file. Will be inferred if not given.
    <any other m.swim.routing argument>, optional
        Any other argument for m.swim.routing will be parsed.
    """
    module = 'm.swim.routing'
    # default module arguments
    accumulation = 'accumulation'
    # get subbasins raster name from Subbasins instance
    subbasins = property(lambda self: self.project.subbasins.vector)

    def __init__(self, project):
        self.project = project
        if not (hasattr(project, 'figpath') or hasattr(self, 'figpath')):
            fp = 'input/%s.fig' % self.project.project_name
            self.figpath = osp.join(project.projectdir, fp)
        return


class Substats(mmgrass.GrassModulePlugin):
    """Plugin to deal with the subbasin statistics SWIM input.

    Subbasins vector, mainstreams, drainage and accumulation raster must exist.

    The following attributes are either needed as project settings,
    class/instance attributes or may be parsed to the create and update method:

    Attributes
    ----------
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    projectname : str, optional
        Name of project. Will be inferred if not given.
    <any other m.swim.substats argument>, optional
        Any other argument for m.swim.substats will be parsed.
    """
    module = 'm.swim.substats'
    # default module arguments
    inputpath = 'input'  # will be expanded
    subbasins = property(lambda self: self.project.subbasins.vector)

    def __init__(self, project):
        self.project = project
        self.projectpath = osp.join(project.projectdir, self.inputpath)
        if not (hasattr(project, 'projectname') or
                hasattr(self, 'projectname')):
            self.projectname = self.project.project_name
        return


class Hydrotopes(mmgrass.GrassModulePlugin):
    """Plugin to deal with hydrotope-related input.

    Subbasins raster must exist.

    The following attributes are either needed as project settings,
    class/instance attributes or may be parsed to the create and update method:

    Attributes
    ----------
    landuse : str
        Name of landuse raster map.
    soil : str
        Name of soil raster map.
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    strfilepath : str, optional
        Path to str file. Will be inferred if not given.
    <any other m.swim.hydrotope argument>, optional
        Any other argument for m.swim.hydrotopes will be parsed.
    """
    module = 'm.swim.hydrotopes'
    #: Hydrotopes raster (w/o mapset) to be created in `grass_mapset`
    raster = 'hydrotopes'
    #: Subbasin raster taken from subbasins.raster
    subbasins = property(lambda self: self.project.subbasins.raster)
    hydrotopes = property(lambda self: self.raster)

    def __init__(self, project):
        self.project = project
        if not (hasattr(project, 'strfilepath') or
                hasattr(self, 'strfilepath')):
            strfilepath = 'input/%s.str' % self.project.project_name
            self.strfilepath = osp.join(project.projectdir, strfilepath)
        return

    def reclass(self, values, outrast, mapset=None):
        """Reclass hydrotopes raster to 'values'.

        Arguments
        ---------
        outrast : str
            Name of raster to be created.
        values : list-like | <pandas.Series>
            Values the raster is reclassed to. If a pandas.Series is parsed,
            the index is used to create the mapping, otherwise the categories
            are assumed to be ``range(1, len(values)+1)``.
        mapset : str, optional
            Different than the default `grass_mapset` mapset to write to.
        """
        hydname = self.raster + '@' + self.project.grass_mapset
        reclass_raster(self.project, hydname, outrast, values, mapset=mapset)
        return


def reclass_raster(project, inrast, outrast, values, mapset=None):
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
    """
    from pandas import Series
    if isinstance(values, Series):
        ix = values.index
    else:
        ix = range(1, len(values)+1)
    columns = np.column_stack([ix, ix, values, values])
    mapset = mapset or project.grass_mapset
    with mmgrass.GrassSession(project, mapset=mapset) as grass:
        tf = grass.tempfile()
        np.savetxt(tf, columns, delimiter=':')
        grass.run_command('r.recode', input=inrast, output=outrast, rules=tf)
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
        Mapset to write to. Default: `prefix` (trailing _ trimmed)
    """
    # argument preparation
    prefix = prefix or values.__class__.__name__
    mapset = mapset or prefix.strip('_')
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
    # do the reclassing
    for i, n in enumerate(names):
        reclasser(d.iloc[i, :], n, mapset=mapset)
        if int(os.environ.get('GRASS_VERBOSE', 1)):
            print('Created raster %s' % (n+'@'+mapset))
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
            print('Created space-time raster dataset %s' % (dsname+'@'+mapset))
    return
