from __future__ import absolute_import
import os.path as osp

from modelmanager import utils as mmutils
from modelmanager.plugins import grass as mmgrass


class Subbasins(mmgrass.GrassModulePlugin):
    """Plugin to deal with subbasin-related input creation.

    Settings
    --------
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

    # default module arguments
    subbasins = 'subbasins'

    def postprocess(self, **moduleargs):
        self.project.routing(**moduleargs)
        self.project.substats(**moduleargs)
        self.project.hydrotopes(**moduleargs)
        return


class Routing(mmgrass.GrassModulePlugin):
    """Plugin to deal with routing-related input.

    Subbasins vector and accumulation raster must exist.

    Settings
    --------
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    figpath : str, optional
        Path to fig file. Will be inferred if not given.
    <any other m.swim.routing argument>, optional
        Any other argument for m.swim.routing will be parsed.

    Note
    ----
    The `subbasins` argument must be given as attribute as giving it as project
    setting would overwrite the subbasins plugin.
    """
    module = 'm.swim.routing'
    # default module arguments
    subbasins = 'subbasins'
    accumulation = 'accumulation'

    def __init__(self, project):
        self.project = project
        if not (hasattr(project, 'figpath') or hasattr(self, 'figpath')):
            ppn = mmutils.get_paths_pattern('input/*.cod', project.projectdir)
            assert len(ppn) == 1, 'figpath not set and cant be inferred.'
            pname = osp.splitext(osp.basename(ppn[0]))[0]
            self.figpath = osp.join(project.projectdir, 'input/%s.fig' % pname)
        return


class Substats(mmgrass.GrassModulePlugin):
    """Plugin to deal with the subbasin statistics SWIM input.

    Subbasins vector, mainstreams, drainage and accumulation raster must exist.

    Settings
    --------
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    projectname : str, optional
        Name of project. Will be inferred if not given.
    <any other m.swim.substats argument>, optional
        Any other argument for m.swim.substats will be parsed.

    Note
    ----
    The `subbasins` argument must be given as attribute as giving it as project
    setting would overwrite the subbasins plugin.
    """
    module = 'm.swim.substats'
    # default module arguments
    subbasins = 'subbasins'
    inputpath = 'input'  # will be expanded

    def __init__(self, project):
        self.project = project
        self.projectpath = osp.join(project.projectdir, self.inputpath)
        if not (hasattr(project, 'projectname') or
                hasattr(self, 'projectname')):
            ppn = mmutils.get_paths_pattern('input/*.cod', project.projectdir)
            assert len(ppn) == 1, 'projectname not set and cant be inferred.'
            self.projectname = osp.splitext(osp.basename(ppn[0]))[0]
        return


class Hydrotopes(mmgrass.GrassModulePlugin):
    """Plugin to deal with hydrotope-related input.

    Subbasins raster must exist.

    Settings
    --------
    landuse : str
        Name of landuse raster map.
    soil : str
        Name of soil raster map.
    <settings required for mmgrass.GrassSession>
        grass_db, grass_location, grass_mapset are needed.
    strfilepath : str, optional
        Path to str file. Will be inferred if not given.
    <any other m.swim.hydrotope argument>, optional
        Any other argument for m.swim.substats will be parsed.

    Note
    ----
    The `subbasins` argument must be given as attribute as giving it as project
    setting would overwrite the subbasins plugin.
    """
    module = 'm.swim.hydrotopes'
    # default module arguments
    subbasins = 'subbasins'
    hydrotopes = 'hydrotopes'

    def __init__(self, project):
        self.project = project
        if not (hasattr(project, 'strfilepath') or
                hasattr(self, 'strfilepath')):
            ppn = mmutils.get_paths_pattern('input/*.cod', project.projectdir)
            assert len(ppn) == 1, 'strfilepath not set and cant be inferred.'
            pname = osp.splitext(osp.basename(ppn[0]))[0]
            strfilepath = 'input/%s.str' % pname
            self.strfilepath = osp.join(project.projectdir, strfilepath)
        return
