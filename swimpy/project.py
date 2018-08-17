# -*- coding: utf-8 -*-

"""
The main project module.
"""
import os
import os.path as osp
from glob import glob
import datetime as dt
import subprocess
from numbers import Number
from decimal import Decimal

import modelmanager as mm
from modelmanager.settings import parse_settings

from swimpy import utils
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
    resourcedir : str path
        Absoulte path to the swimpy resource directory.
    project_name : str
        Project name inferred from the ``input/*.cod`` file.
    settings : modelmanager.SettingsManager
        A manager object that is used to attach, record and check settings.
    """

    def __init__(self, projectdir='.', **settings):
        # load default settings
        defaults = mm.settings.load_settings(defaultsettings)
        for k, v in defaults.items():
            settings.setdefault(k, v)
        # check for project name
        settings.setdefault('project_name', self._get_project_name(projectdir))
        # load modelmanager Project
        super(Project, self).__init__(projectdir, **settings)
        self.settings.defaults = defaults
        return

    def _get_project_name(self, projectdir=None):
        """Infer the project name from the code file."""
        ppn = glob(osp.join(projectdir or self.projectdir, 'input/*.cod'))
        if len(ppn) == 1:
            return osp.splitext(osp.basename(ppn[0]))[0]
        return None

    @parse_settings
    def run(self, save=True, cluster=False, quiet=False, **kw):
        """
        Execute SWIM.

        Arguments
        ---------
        save : bool
            Run save_run after successful execution of SWIM.
        cluster : False | str
            False or a job name to submit this run to SLURM.
        quiet : bool
            Dont show SWIM output if True.
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
            assert type(cluster) is str, "cluster must be a string."
            kw['save'] = save
            return self.submit_cluster(cluster, 'run', **kw)

        swimcommand = [self.swim, self.projectdir+'/']
        # silence output
        stdout = open(os.devnull, 'w') if quiet else None
        # run
        subprocess.check_call(swimcommand, stdout=stdout)
        # save
        if save:
            run = self.save_run(**kw)
        else:
            run = None
        # report runtime
        if not quiet:
            delta = dt.datetime.now() - st
            print('Execution took %s hh:mm:ss' % delta)
        return run

    @parse_settings
    def submit_cluster(self, jobname, functionname=None, script=None,
                       dryrun=False, slurmargs={}, **funcargs):
        """
        Run a project function (method) by submitting it to SLURM.

        Arguments
        ---------
        jobname : str
            SLURM job name.
        functionname : str, optional
            A name string of a project function.
        script : str, optional
            Valid python code to run.
        dryrun : bool
            If True, only write jobfile to cluster resourcedir.
        slurmargs : dict
            SLURM arguments to use for this run temporarily extending /
            overwriting the project slurmargs attribute.
        **funcargs : optional
            Arguments parsed to function.

        Returns
        -------
        int
            The job ID.
        """
        assert type(functionname) == str or type(script) == str
        if functionname:
            assert callable(self.settings[functionname])
            script = ("import swimpy; p=swimpy.Project(); p.%s(**%r)" %
                      (functionname, funcargs))
        # dir for slurm job, output, error files
        outdir = osp.join(self.resourcedir, 'cluster')
        if not osp.exists(outdir):
            os.mkdir(outdir)
        # submit to slurm
        for k, v in self.slurmargs.items():
            slurmargs.setdefault(k, v)
        rid = utils.slurm_submit(jobname, script, outdir, dryrun=dryrun,
                                 workdir=self.projectdir, **slurmargs)
        return rid

    def __call__(self, *runargs, **runkwargs):
        """
        Shortcut for run(); see run documentation.
        """
        return self.run(*runargs, **runkwargs)

    def save_resultindicator(self, run, name, value, tags=''):
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
        ResultIndicator (Django model) instance or list of instances.
        """
        def insert_ind(**kwargs):
            return self.browser.insert('resultindicator', **kwargs)
        emsg = ('Indicator %s is not a number ' % name +
                'or a dictionary of numbers. Instead: %r' % value)
        if isinstance(value, Number):
            i = insert_ind(run=run, name=name, value=value, tags=tags)
        elif type(value) is dict:
            assert all([isinstance(v, Number) for v in value.values()]), emsg
            i = [insert_ind(run=run, name=name, value=v, tags=tags+' '+str(k))
                 for k, v in value.items()]
        else:
            raise IOError(emsg)
        return i

    def save_resultfile(self, run, tags, filelike):
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
        ResultFile (Django model) instance or list of instances.
        """
        errmsg = ('%s is not a file instance, existing path or ' % filelike +
                  'pandas DataFrame/Series or dictionary of those.')

        def is_valid(fu):
            flik = all([hasattr(fu, m) for m in ['read', 'close', 'seek']])
            return flik or (type(fu) is str and osp.exists(fu))

        def insert_file(**kwargs):
            return self.browser.insert('resultfile', **kwargs)

        if hasattr(filelike, 'to_run'):
            f = filelike.to_run(run)
        elif hasattr(filelike, 'to_csv'):
            fn = '_'.join(tags.split())+'.csv.gzip'
            tmpf = osp.join(self.browser.settings.tmpfilesdir, fn)
            filelike.to_csv(tmpf, compression='gzip')
            f = insert_file(run=run, tags=tags, file=tmpf)
        elif type(filelike) == dict:
            assert all([is_valid(v) for v in filelike.values()]), errmsg
            f = [self.save_resultfile(run, tags+' '+str(k), v)
                 for k, v in filelike.items()]
        elif is_valid(filelike):
            f = insert_file(run=run, tags=tags, file=filelike)
        else:
            raise IOError(errmsg)
        return f

    @property
    def resultfile_interfaces(self):
        """List of output file project or run attributes.

        Apart from interfacing between current SWIM output files, these
        attributes may be parsed to the `files` argument to `save_run`. They
        will then become an attribute of that run.
        """
        from modelmanager.plugins.pandas import ProjectOrRunData
        fi = [n for n, p in self.settings.properties.items()
              if hasattr(p, 'plugin') and ProjectOrRunData in p.plugin.__mro__]
        return fi

    @parse_settings
    def save_run(self, indicators={}, files={}, **kw):
        """
        Save the current SWIM input/output as a run in the browser database.

        Arguments
        ---------
        indicators : dict | list
            Dictionary of indicator values passed to self.save_resultindicator
            or list of method or attribute names that return an indicator
            (float) or dictionary of those.
        files : dict | list
            Dictionary of file values passed to self.save_resultfile or list of
            method or attribute names that return any of file instance, a file
            path or a pandas.DataFrame/Series (will be converted to file via
            to_csv) or a dictionary of any of those.
        **kw : optional
            Set fields of the run browser table. Default fields: notes, tags.

        Returns
        -------
        Run object (Django model object).
        """
        assert type(indicators) in [list, dict]
        assert type(files) in [list, dict]
        # config
        sty, nbyr = self.config_parameters('iyr', 'nbyr')
        run_kwargs = {'start': dt.date(sty, 1, 1),
                      'end': dt.date(sty + nbyr - 1, 12, 31),
                      'parameters': self.changed_parameters()}
        run_kwargs.update(kw)
        # create run
        run = self.browser.insert('run', **run_kwargs)

        # add files and indicators
        for tbl, a in [('resultindicator', indicators), ('resultfile', files)]:
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


def setup(projectdir='.', resourcedir='swimpy'):
    """
    Setup a swimpy project.

    Arguments
    ---------
    projectdir : str path
        Project directory. Will be created if not existing.
    resourcedir : str
        Name of swimpy resource directory in projectdir.

    Returns
    -------
    Project instance.
    """
    mmproject = mm.project.setup(projectdir, resourcedir)
    # swim specific customisation of resourcedir
    defaultsdir = osp.join(osp.dirname(__file__), 'resources')
    mm.utils.copy_resources(defaultsdir, mmproject.resourcedir, overwrite=True)
    # FIXME rename templates with project name in filename
    for fp in ['cod', 'bsn']:
        ppn = mm.utils.get_paths_pattern('input/*.' + fp, projectdir)
        tp = osp.join(mmproject.resourcedir, 'templates')
        if len(ppn) > 0:
            os.rename(osp.join(tp, 'input/%s.txt' % fp), osp.join(tp, ppn[0]))
    # load as a swim project
    project = Project(projectdir)
    return project
