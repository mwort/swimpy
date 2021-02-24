# -*- coding: utf-8 -*-

"""
The main project module.
"""
from __future__ import absolute_import, print_function
import os
import os.path as osp
import sys
import glob
import shutil
import datetime as dt
import subprocess
from numbers import Number
from decimal import Decimal

import modelmanager as mm
from modelmanager.settings import SettingsManager, parse_settings

from swimpy import defaultsettings


class Project(mm.Project):
    """The project object with settings attached.

    All settings are available as attributes, methods, properties and plugin
    instances (i.e. also attributes) from the project instance. By default
    projects have all attributes listed in :mod:`swimpy.defaultsettings` as
    well as the following:

    Attributes
    ----------
    projectdir : str path
        Absolute path to the project directory.
    resourcedir : bool
        Assume resourcedir to exist or load project without resources.
    settings : modelmanager.SettingsManager
        A manager object that is used to attach, record and check settings.
    """

    def __init__(self, projectdir='.', resourcedir=True, **settings):
        self.projectdir = osp.abspath(projectdir)
        self.settings = SettingsManager(self)
        # load default settings
        self.settings.defaults = mm.settings.load_settings(defaultsettings)
        if not resourcedir:
            for s in defaultsettings.plugins_require_resourcedir:
                self.settings.defaults.pop(s, None)
        # load settings with overridden settings
        self.settings.load(defaults=self.settings.defaults, resourcedir=True,
                           **settings)
        return

    @parse_settings
    def run(self, save=True, cluster=None, quiet=False, **kw):
        """
        Execute SWIM.

        Arguments
        ---------
        save : bool
            Run save_run after successful execution of SWIM.
        cluster : str | dict
            Job name to submit this run to SLURM or a dict will set
            other cluster() arguments but must include a ``jobname``.
        quiet : bool | str path
            Dont show SWIM output if True or redirect it to a file path.
        **kw : optional
            Keyword arguments passed to save_run.

        Returns
        --------
        Run instance (Django model object) or  None if save=False.
        """
        # starting clock
        st = dt.datetime.now()
        # if submitting to cluster
        if cluster:
            kw.update({'functionname': 'run', 'save': save, 'quiet': quiet})
            return self.cluster(cluster, **kw)

        swimcommand = [self.swim, self.projectdir+'/']
        # silence output
        sof = quiet if type(quiet) == str else os.devnull
        stdout = open(sof, 'w') if quiet else None
        # run
        subprocess.check_call(swimcommand, stdout=stdout)
        if quiet:
            stdout.close()
        delta = dt.datetime.now() - st
        # save
        if save:
            run = self.save_run(run_time=delta, **kw)
        else:
            run = None
        # report runtime
        if not quiet:
            print('Execution took %s hh:mm:ss' % delta)
        return run

    def __call__(self, *runargs, **runkwargs):
        """
        Shortcut for run(); see run documentation.
        """
        return self.run(*runargs, **runkwargs)

    def save_indicator(self, run, name, value, tags=''):
        """
        Save a result indicator with a run.

        Arguments
        ---------
        run : models.Run instance | int
            SWIM run object or an ID.
        name : str
            Name of indicator.
        value : number or dict of numbers
            Number or dictionary of indicator values (float/int will be
            converted to Decimal).
        tags : str
            Additional tags (space separated).

        Returns
        -------
        indicator (Django model) instance or list of instances.
        """
        def insert_ind(**kwargs):
            return self.browser.insert('indicator', **kwargs)
        emsg = ('Indicator %s is not a number ' % name +
                'or a dictionary of numbers. Instead: %r' % value)
        if isinstance(value, Number):
            i = insert_ind(run=run, name=name, value=value, tags=tags)
        elif type(value) is dict or hasattr(value, 'items'):
            assert all([isinstance(v, Number) for k, v in value.items()]), emsg
            i = [insert_ind(run=run, name=name, value=v, tags=tags+' '+str(k))
                 for k, v in value.items()]
        else:
            raise IOError(emsg)
        return i

    def save_file(self, run, tags, filelike):
        """
        Save a result file with a run.

        Arguments
        ---------
        run : Run instance (Django model) | int
        tags : str
            Space-separated tags. Will be used as file name if pandas objects
            are parsed.
        filelike : file-like object | pandas.Dataframe/Series | dict of those
            A file instance, a file path or a pandas.DataFrame/Series
            (will be converted to file via to_csv) or a dictionary of any of
            those types (keys will be appended to tags).

        Returns
        -------
        browser.File (Django model) instance or list of instances.
        """
        errmsg = ('%s is not a file instance, existing path or ' % filelike +
                  'pandas DataFrame/Series or dictionary of those.')

        def is_valid(fu):
            flik = all([hasattr(fu, m) for m in ['read', 'close', 'seek']])
            return flik or (type(fu) is str and osp.exists(fu))

        def insert_file(**kwargs):
            return self.browser.insert('file', **kwargs)

        if hasattr(filelike, 'to_run'):
            f = filelike.to_run(run, tags=tags)
        elif hasattr(filelike, 'to_csv'):
            tmpdir = osp.join(self.browser.settings.tmpfilesdir, 'runs',
                              str(getattr(run, 'pk', run)))
            try:
                os.makedirs(tmpdir)
            except OSError:
                pass
            tmpf = osp.join(tmpdir, '_'.join(tags.split())+'.csv.gzip')
            filelike.to_csv(tmpf, compression='gzip')
            f = insert_file(run=run, tags=tags, file=tmpf, copy=False)
        elif type(filelike) == dict:
            assert all([is_valid(v) for v in filelike.values()]), errmsg
            f = [self.save_file(run, tags+' '+str(k), v)
                 for k, v in filelike.items()]
        elif is_valid(filelike):
            f = insert_file(run=run, tags=tags, file=filelike)
        else:
            raise IOError(errmsg)
        return f

    @property
    def output_interfaces(self):
        """List of output file project or run attributes.

        Apart from interfacing between current SWIM output files, these
        attributes may be parsed to the `files` argument to `save_run`. They
        will then become an attribute of that run.
        """
        from modelmanager.plugins.pandas import ProjectOrRunData
        fi = [n for n, p in self.settings.properties.items()
              if hasattr(p, 'plugin') and ProjectOrRunData in p.plugin.__mro__]
        return fi

    def output_interface_paths(self, print_=False):
        """Return dict (or print) of names and absolute (relative) paths."""
        oi = {}
        for rf in self.output_interfaces:
            pth = self.settings.properties[rf].plugin.path
            if pth:
                oi[rf] = osp.join(self.projectdir, pth)
        if print_:
            for n, p in oi.items():
                print('%s: %s' % (n, osp.relpath(p, os.getcwd())))
        else:
            return oi

    @parse_settings
    def save_run(self, indicators=None, files=None, parameters=True, **kw):
        """
        Save the current SWIM input/output as a run in the browser database.

        Arguments
        ---------
        indicators : dict | list
            Dictionary of indicator values passed to self.save_indicator
            or list of method or attribute names that return an indicator
            (float) or dictionary of those.
        files : dict | list
            Dictionary of file values passed to self.save_file or list of
            method or attribute names that return any of file instance, a file
            path or a pandas.DataFrame/Series (will be converted to file via
            to_csv) or a dictionary of any of those.
        parameters : list of dicts | bool
            Save parameter changes with the run. The dicts must contain the
            parameter table attributes attributes. If True (default) use result
            of ``self.changed_parameters()``, if False dont save any.
        **kw : optional
            Set fields of the run browser table. Default fields: notes, tags.

        Returns
        -------
        Run object (Django model object).
        """
        indicators = indicators or {}
        files = files or {}
        assert type(indicators) in [list, dict]
        assert type(files) in [list, dict]
        # config
        sty, nbyr = self.config_parameters('iyr', 'nbyr')
        run_kwargs = {'start': dt.date(sty, 1, 1),
                      'end': dt.date(sty + nbyr - 1, 12, 31)}
        if parameters:
            if parameters is True:
                parameters = self.changed_parameters()
            run_kwargs['parameters'] = parameters

        run_kwargs.update(kw)
        # create run
        run = self.browser.insert('run', **run_kwargs)

        # add files and indicators
        for tbl, a in [('indicator', indicators), ('file', files)]:
            save_function = getattr(self, 'save_' + tbl)
            # unpack references
            if type(a) == list:
                a = {k: self._attribute_or_function_result(k) for k in a}
            for n, m in a.items():
                save_function(run, n, m)
        return run

    def _attribute_or_function_result(self, m):
        try:
            fv = self.settings[m]
            if callable(fv):
                fv = fv()
        except Exception as e:
            print(e)
            raise Exception('Failed to call function %s' % m)
        return fv

    def changed_parameters(self, verbose=False):
        """
        Compare currently set basin and subcatch parameters with the last in
        the parameter browser table.

        Arguments
        ---------
        verbose : bool
            Print changes.

        Returns
        -------
        list
            List of dictionaries with parameter browser attributes.
        """
        changed = []
        # create dicts with (pnam, stationID): value
        bsnp = self.basin_parameters
        scp = self.subcatch_parameters.T.stack().to_dict()
        for k, v in list(bsnp.items()) + list(scp.items()):
            n, sid = k if type(k) == tuple else (k, None)
            # convert to minimal precision decimal via string
            dv = Decimal(str(v))
            saved = self.browser.parameters.filter(name=n, tags=sid).last()
            if not saved or saved.value != dv:
                changed += [dict(name=n, value=v, tags=sid)]
                if verbose:
                    sv = saved[-1]['value'] if saved else None
                    print('%s: %s > %s' % (k, sv, dv))
        return changed


def setup(projectdir='.', name=None, gitrepo=None, resourcedir='swimpy'):
    """
    Setup a swimpy project.

    If the SWIM input and output directories are not existing in projectdir,
    they will be created and the SWIM repository's ``project`` is used to setup
    new input files. In that case, name and repo will be prompted if they are
    not parsed already.

    Arguments
    ---------
    projectdir : str path
        Project directory. Will be created if not existing.
    name : str
        Name of the new project, if input directory doesnt exists.
    gitrepo : str
        URL or path to the SWIM repository. If a URL is given, the repository
        is cloned into the resourcedir. (Defaults to swimpy.swim_url)
    resourcedir : str
        Name of swimpy resource directory in projectdir.

    Returns
    -------
    Project instance.
    """
    # defaults
    import swimpy
    swim_url = swimpy.swim_url
    dirs = ['input', 'output/Res', 'output/GIS', 'output/Flo']

    mmproject = mm.project.setup(projectdir, resourcedir)
    # swim specific customisation of resourcedir
    defaultsdir = osp.join(osp.dirname(__file__), 'resources')
    mm.utils.copy_resources(defaultsdir, mmproject.resourcedir, overwrite=True)

    # gitrepo and name needed, define testinputpath
    if not osp.exists(osp.join(projectdir, 'input')):
        inp = input if sys.version_info.major > 2 else raw_input
        name = name or inp('Enter project name: ').lower()
        gitrepo = gitrepo or swim_url
        if osp.exists(gitrepo):
            repopath = gitrepo
        else:
            repopath = osp.join(mmproject.resourcedir, 'swim')
            subprocess.check_call(['git', 'clone', '-q', gitrepo, repopath])
    else:
        repopath = None

    # create empty directories as needed
    for d in dirs:
        dp = osp.join(projectdir, d)
        if not osp.exists(dp):
            os.makedirs(dp)

    # copy test project files
    if repopath:
        testinputpath = osp.join(repopath, 'project', 'input')
        print('Copying all SWIM default input files from: %s' % testinputpath)
        mm.utils.copy_resources(testinputpath, osp.join(projectdir, 'input'))
        for f in glob.glob(osp.join(projectdir, 'input', '*blank*')):
            newf = osp.basename(f).replace('blank', name)
            os.rename(f, osp.join(osp.dirname(f), newf))
        # link swim exe and conf
        os.symlink(osp.join(repopath, 'code', 'swim'),
                   osp.join(projectdir, 'swim'))
        shutil.copy(osp.join(repopath, 'project', 'swim.conf'), projectdir)
        # change input names in file.cio
        with open(osp.join(projectdir, 'input', 'file.cio')) as f:
            cio = f.read().replace('blank', name)
        with open(osp.join(projectdir, 'input', 'file.cio'), 'w') as f:
            f.write(cio.replace('blank', name))
    # rename templates with project name in filename
    for fp in ['cod', 'bsn']:
        ppn = mm.utils.get_paths_pattern('input/*.' + fp, projectdir)
        tp = osp.join(mmproject.resourcedir, 'templates')
        if len(ppn) > 0:
            os.rename(osp.join(tp, 'input/%s.txt' % fp), osp.join(tp, ppn[0]))
    # load as a swim project
    project = Project(projectdir)
    return project
