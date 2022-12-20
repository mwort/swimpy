# -*- coding: utf-8 -*-

"""
The main project module.
"""
from __future__ import absolute_import, print_function
import os
import os.path as osp
import sys
from warnings import warn
from glob import glob
import shutil
import datetime as dt
import subprocess
from numbers import Number
from decimal import Decimal

import modelmanager as mm
from modelmanager.settings import SettingsManager, parse_settings
from modelmanager.project import ProjectDoesNotExist

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
        Absolute path to the project directory or <parameters>.nml.
    resourcedir : bool
        Assume resourcedir to exist or load project without resources.
    settings : modelmanager.SettingsManager
        A manager object that is used to attach, record and check settings.
    """

    def __init__(self, projectdir='.', resourcedir=True, **settings):
        if osp.isdir(projectdir):
            self.projectdir = osp.abspath(projectdir)
            parfile = glob(osp.join(self.projectdir, '*.nml'))
            if len(parfile) == 0:
                raise ProjectDoesNotExist("Could not find a parameter file *.nml in given directory!")
            if len(parfile) > 1:
                raise ProjectDoesNotExist('Found more than one *.nml in given directory. Please specify a single *.nml!')
            self.parfile = osp.basename(parfile[0])
        elif osp.isfile(projectdir):
            self.parfile = osp.basename(projectdir)
            self.projectdir = osp.abspath(osp.dirname(projectdir))
        else:
            raise ProjectDoesNotExist('Input projectdir must be a directory or file path!')
        self.settings = SettingsManager(self)
        # load default settings
        self.settings.defaults = mm.settings.load_settings(defaultsettings)
        if not resourcedir:
            for s in defaultsettings.plugins_require_resourcedir:
                self.settings.defaults.pop(s, None)
        # load settings with overridden settings
        self.settings.load(defaults=self.settings.defaults, resourcedir=True,
                           **settings)
        # create input and output dirs if they do not exist
        if not osp.exists(self.inputpath):
            os.makedirs(self.inputpath, exist_ok=True)
        if not osp.exists(self.outputpath):
            os.makedirs(self.outputpath, exist_ok=True)
        return
    
    @property
    def inputpath(self):
        path = osp.join(self.projectdir,
                        self.config_parameters['input_dir'])
        return self._path if hasattr(self, '_path') else path

    @inputpath.setter
    def inputpath(self, value):
        self._path = value
        return

    @property
    def outputpath(self):
        path = osp.join(self.projectdir,
                        self.config_parameters['output_dir'])
        return self._path if hasattr(self, '_path') else path

    @outputpath.setter
    def outputpath(self, value):
        self._path = value
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
        # swim command
        ppn = glob(osp.join(self.projectdir, self.parfile))[0]
        swimcommand = [self.swim, ppn]
        # silence output
        sof = quiet if type(quiet) == str else os.devnull
        stdout = open(sof, 'w') if quiet else None
        # run
        subprocess.check_call(swimcommand, stdout=stdout, cwd=self.projectdir)
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
        warn('project.output_interfaces is deprecated and will be removed in '
            'a future version. Use project.output_files instead.',
            FutureWarning, stacklevel=2)
        return self.output_files

    def output_interface_paths(self, print_=False):
        warn('project.output_interface_paths is deprecated and will be removed in '
            'a future version. Use project.outputpath instead.',
            FutureWarning, stacklevel=2)
        """Return dict (or print) of names and absolute (relative) paths."""
        if print_:
            for n in self.output_files.keys():
                print('%s: %s' % (n+'.csv', self.output_parameters['output_dir']))
        else:
            return self.outputpath

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
            parameter table attributes. If True (default) use result
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
        run_kwargs = {'start': self.config_parameters.start_date,
                      'end': self.config_parameters.end_date}
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
        Compare currently set config_parameters and catchment parameters with
        the last in the parameter browser table.

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
        bsnp = self.config_parameters.parlist
        scp = self.catchment.T.stack().to_dict()
        for k, v in list(bsnp.items()) + list(scp.items()):
            n, sid = k if type(k) == tuple else (k, None)
            # convert to minimal precision decimal via string
            v = v if type(v) != bool else int(v)
            # TODO: enable string parameters, see
            # models.py here and/or in modelmanager
            # dv = Decimal(str(v))
            if isinstance(v, Number):
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
        Name of the new project, if input directory does not exist.
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

    # copy test project files
    if repopath:
        testinputpath = osp.join(repopath, 'project', 'input')
        print('Copying all SWIM default input files from: %s' % testinputpath)
        mm.utils.copy_resources(testinputpath, osp.join(projectdir, 'input'))
        # link swim exe
        os.symlink(osp.abspath(osp.join(repopath, 'code', 'swim')),
                   osp.abspath(osp.join(projectdir, 'swim')))
        # default parameters .nml
        shutil.copyfile(osp.join(repopath, 'project/blankenstein_parameters.nml'),
                        osp.join(projectdir, name+'.nml'))
    # load as a swim project
    project = Project(projectdir)
    return project
