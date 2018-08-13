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
import os.path as osp

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
    vector = 'subbasins'
    subbasins = property(lambda self: self.vector)

    def postprocess(self, **moduleargs):
        self.project.routing(**moduleargs)
        self.project.substats(**moduleargs)
        self.project.hydrotopes(**moduleargs)
        return

    @propertyplugin
    class attributes(mmgrass.GrassAttributeTable):
        """The subbasins attribute table as a ``pandas.DataFrame`` object."""
        vector = property(lambda self: self.project.subbasins.vector)


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
    raster = 'hydrotopes'
    subbasins = property(lambda self: self.project.subbasins.vector)
    hydrotopes = property(lambda self: self.raster)

    def __init__(self, project):
        self.project = project
        if not (hasattr(project, 'strfilepath') or
                hasattr(self, 'strfilepath')):
            strfilepath = 'input/%s.str' % self.project.project_name
            self.strfilepath = osp.join(project.projectdir, strfilepath)
        return
