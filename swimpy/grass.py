from __future__ import absolute_import
import os
import os.path as osp
import sys
import subprocess

from modelmanager import utils as mmutils


class GrassSession(object):
    """Open a GRASS session in a mapset without launching GRASS.

    To be used as a context. The context variable is the grass.script module:
    with GrassSession('path/to/mapset') as grass:
        grass.run_command()

    Arguments
    ---------
    gisdb : str path
        Path to the GRASS database or straight to the mapset if location
        and mapset are None.
    location, mapset : str, optional
    grassbin : str
        Name or path of GRASS executable.
    """
    def __init__(self, gisdb, location=None, mapset=None, grassbin='grass'):
        # if gisdb is path to mapset
        assert osp.exists(gisdb), 'gisdb doesnt exist: %s' % gisdb
        if not location and not mapset:
            gisdb = gisdb[:-1] if gisdb.endswith('/') else gisdb
            mapset = os.path.basename(gisdb)
            dblo = os.path.dirname(gisdb)
            location = os.path.basename(dblo)
            gisdb = os.path.dirname(dblo)
        errmsg = 'location %s doesnt exist.' % location
        assert osp.exists(osp.join(gisdb, location)), errmsg
        self.gisdb, self.location, self.mapset = gisdb, location, mapset
        # query GRASS 7 itself for its GISBASE
        errmsg = "%s not found or not executable." % grassbin
        assert self._which(grassbin), errmsg
        startcmd = [grassbin, '--config', 'path']
        p = subprocess.Popen(startcmd, shell=False, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise ImportError("ERROR: Cannot find GRASS GIS 7 start script "
                              "using %s. Try passing correct grassbin."
                              % (' '.join(startcmd)))
        self.gisbase = out.strip().split('\n')[-1]
        userdir = osp.expanduser("~")
        self.addonpath = osp.join(userdir, '.grass7', 'addons', 'scripts')
        self.python_package = os.path.join(self.gisbase, "etc", "python")
        return

    def _which(self, program):
        fpath, fname = os.path.split(program)
        if fpath:
            if os.path.isfile(program) and os.access(program, os.X_OK):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, program)
                if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
                    return exe_file
        return None

    def setup(self):
        # Set environment variables
        os.environ['GISBASE'] = self.gisbase
        os.environ['GISDBASE'] = self.gisdb
        # add path to GRASS addons depends on build/platform
        os.environ['PATH'] += os.pathsep + self.addonpath
        sys.path.insert(0, self.python_package)

        import grass.script as grass
        from grass.script import setup

        self.grass = grass
        self.rcfile = setup.init(self.gisbase, self.gisdb,
                                 self.location, 'PERMANENT')
        # always create mapset if it doesnt exist
        grass.run_command('g.mapset', mapset=self.mapset, flags='c', quiet=1)
        return grass

    def __enter__(self):
        return self.setup()

    def clean(self):
        # remove .gislock and rc file if exists
        env = self.grass.gisenv()
        lf = osp.join(env['GISDBASE'], env['LOCATION_NAME'], env['MAPSET'],
                      '.gislock')
        [os.remove(f) for f in [lf, self.rcfile] if osp.exists(lf)]
        # clean envs
        sys.path = [p for p in sys.path if p is not self.python_package]
        ps = os.pathsep
        rmpaths = [self.addonpath]
        paths = os.environ['PATH'].split(ps)
        os.environ['PATH'] = ps.join([p for p in paths if p not in rmpaths])
        return

    def __exit__(self, *args):
        self.clean()
        return


class ProjectGrassSession(GrassSession):
    """Grass session for the swimpy project.

    Settings
    --------
    grass_db : str path
        Path to grass database.
    grass_location : str
        Grass location name.
    grass_mapset : str
        Grass mapset where to write to.
    grassbin : str, optional (default=grass)
        Name or path of the grass executable.
    """
    def __init__(self, project, **override):
        self.project = project
        kw = dict(gisdb=project.grass_db,
                  location=project.grass_location,
                  mapset=project.grass_mapset)
        if hasattr(project, "grassbin"):
            kw['grassbin'] = project.grassbin
        kw.update(override)
        super(ProjectGrassSession, self).__init__(**kw)
        return


class GrassModulePlugin(object):
    """A representation of a grass module that takes arguments from either
    project settings or its own attributes.
    """

    module = None

    def __init__(self, project):
        self.project = project
        errmsg = '%s.module has to be set.' % self.__class__.__name__
        assert self.module, errmsg
        return

    def create(self, **moduleargs):
        """Run the related grass module.

        Arguments
        ---------
        **moduleargs :
            Override any arguments of the module alredy set in settings.
        """
        args = {}
        with ProjectGrassSession(self.project):
            from grass.pygrass.modules import Module
            module = Module(self.module, run_=False)
            for p in module.params_list:
                if p.name in moduleargs:
                    args[p.name] = moduleargs[p.name]
                elif hasattr(self, p.name):
                    args[p.name] = getattr(self, p.name)
                elif hasattr(self.project, p.name):
                    args[p.name] = getattr(self.project, p.name)
                elif p.required:
                    em = p.name + ' argument is required by ' + self.module
                    raise AttributeError(em)
            # run module
            module(**args).run()
        return

    def update(self, **modulekwargs):
        """Run create and postprocess with GRASS_OVERWRITE."""
        GO = 'GRASS_OVERWRITE'
        is_set = GO in os.environ and os.environ[GO] != '0'
        os.environ[GO] = '1'
        self.create(**modulekwargs)
        self.postprocess(**modulekwargs)
        if not is_set:
            os.environ.pop(GO)
        return

    def __call__(self, **modulekwargs):
        """Shortcut for `update`."""
        return self.update(**modulekwargs)

    def postprocess(self):
        """Overwrite to perform follow up tasks."""
        return


class Subbasins(GrassModulePlugin):
    """Plugin to deal with subbasin-related input creation.

    Settings
    --------
    elevation : str
        Elevation raster name.
    stations : str
        Point vector of gauging stations.
    <settings required for ProjectGrassSession>
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


class Routing(GrassModulePlugin):
    """Plugin to deal with routing-related input.

    Subbasins vector and accumulation raster must exist.

    Settings
    --------
    <settings required for ProjectGrassSession>
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


class Substats(GrassModulePlugin):
    """Plugin to deal with the subbasin statistics SWIM input.

    Subbasins vector, mainstreams, drainage and accumulation raster must exist.

    Settings
    --------
    <settings required for ProjectGrassSession>
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


class Hydrotopes(GrassModulePlugin):
    """Plugin to deal with hydrotope-related input.

    Subbasins raster must exist.

    Settings
    --------
    landuse : str
        Name of landuse raster map.
    soil : str
        Name of soil raster map.
    <settings required for ProjectGrassSession>
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
