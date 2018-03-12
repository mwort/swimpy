# -*- coding: utf-8 -*-

"""
The main project module.
"""
import os
import os.path as osp
import datetime as dt
import subprocess
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from numbers import Number
from decimal import Decimal

import pandas as pa

import modelmanager

from swimpy import utils
from swimpy import results


class Project(modelmanager.Project):

    def __init__(self, projectdir='.', **settings):
        allsettings = {}
        # attach results as propertyplugins
        self.results = modelmanager.utils.load_settings(results)
        projectorrunproperties = self.results.pop('properties')
        allsettings.update(self.results)
        # make sure ProjectOrRunData classes are loaded as properties
        allsettings.update(projectorrunproperties)
        allsettings.update(settings)
        super(Project, self).__init__(projectdir, **allsettings)
        return

    def basin_parameters(self, *getvalues, **setvalues):
        """
        Set or get any values from the .bsn file by variable name.
        """
        pat = 'input/*.bsn'
        if getvalues or setvalues:
            result = self.templates(templates=pat, *getvalues, **setvalues)
        # get all values if no args
        else:
            result = self.templates[pat].read_values()
        return result

    def config_parameters(self, *getvalues, **setvalues):
        """
        Set or get any values from the .cod or swim.conf file by variable name.
        """
        pat = ['input/*.cod', 'swim.conf']
        if getvalues or setvalues:
            result = self.templates(templates=pat, *getvalues, **setvalues)
        else:  # get all values if no args
            result = self.templates[pat[0]].read_values()
            result.update(self.templates[pat[1]].read_values())
        return result

    def subcatch_parameters(self, *getvalues, **setvalues):
        '''
        Read or write parameters in the subcatch.bsn file.

        Reading:
        --------
        (<param> or <stationID>): returns pd.Series
        (<list of param/stationID>): returns subset pa.DataFrame
        (): returns entire table as pd.DataFrame

        Writing:
        --------
        (<param>=value, <stationID>=<list-like>): Assign values or list to
            parameter column or stationID row. Is inserted if not existent.
        (<param>={<stationID>: value, ...}): Set individual values.
        (<pd.DataFrame>): Override entire table, DataFrame must have stationID
            index.
        '''
        filepath = self.subcatch_parameter_file
        # read subcatch.bsn
        bsn = pa.read_table(filepath, delim_whitespace=True)
        stn = 'stationID' if 'stationID' in bsn.columns else 'station'
        bsn.set_index(stn, inplace=True)
        is_df = len(getvalues) == 1 and isinstance(getvalues[0], pa.DataFrame)

        if setvalues or is_df:
            if is_df:
                bsn = getvalues[0]
            for k, v in setvalues.items():
                ix = slice(None)
                if type(v) == dict:
                    ix, v = zip(*v.items())
                if k in bsn.columns:
                    bsn.loc[ix, k] = v
                else:
                    bsn.loc[k, ix] = v
            # write table again
            bsn['stationID'] = bsn.index
            strtbl = bsn.to_string(index=False, index_names=False)
            open(filepath, 'w').write(strtbl)
            return

        if getvalues:
            if all([k in bsn.index for k in getvalues]):
                return bsn.loc[getvalues]
            elif all([k in bsn.columns for k in getvalues]):
                ix = getvalues[0] if len(getvalues) == 1 else list(getvalues)
                return bsn[ix]
            else:
                raise KeyError('Cant find %s in either paramter or stations'
                               % getvalues)
        return bsn

    def run(self, save=True, cluster=False, **save_run_kwargs):
        """
        Execute SWIM.

        Arguments:
        ----------
        save:
            Run save_run after successful execution of SWIM.
        cluster:
            False or a job name to submit this run to SLURM.
        **save_run_kwargs:
            Keyword arguments passed to save_run.
        """
        # starting clock
        st = dt.datetime.now()
        # if submitting to cluster
        if cluster:
            assert type(cluster) is str, "cluster must be a string."
            self.submit_cluster(cluster, 'run', **save_run_kwargs)
            return

        swimcommand = [self.swim, self.projectdir+'/']
        # run
        subprocess.check_call(swimcommand)
        # save
        if save:
            run = self.save_run(**save_run_kwargs)
        else:
            run = None
        # report runtime
        delta = dt.datetime.now() - st
        print('Execution took %s hh:mm:ss' % delta)
        return run

    def submit_cluster(self, jobname, functionname, dryrun=False, **funcargs):
        """
        Run a project function (method) by submitting it to SLURM.

        functionname:
            A name string of a project function.
        jobname:
            SLURM job name.
        dryrun:
            If True, only write jobfile to cluster resourcedir.
        **funcargs:
            Arguments parsed to function.
        """
        assert callable(self.settings[functionname])
        script = ("import swimpy; p=swimpy.Project(); p.%s(**%r)" %
                  (functionname, funcargs))
        # dir for slurm job, output, error files
        outdir = osp.join(self.resourcedir, 'cluster')
        if not osp.exists(outdir):
            os.mkdir(outdir)
        # submit to slurm
        utils.slurm_submit(jobname, script, outdir, dryrun=dryrun,
                           workdir=self.projectdir, **self.slurmargs)
        return

    def __call__(self, *runargs, **runkwargs):
        """
        Shortcut for run(); check run documentation.
        """
        return self.run(*runargs, **runkwargs)

    def save_resultindicator(self, run, name, value, tags=''):
        """
        Save a result indicator with a run.

        run: Django run object or ID.
        name: String name of indicator.
        value: Number or dictionary of indicators (float/int will be converted
            to Decimal).
        tags: Additional tags (space separated).

        Returns: ResultIndicator (Django model) instance or list of instances.
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

    def save_resultfile(self, run, tags, value):
        """
        Save a result file with a run.

        run: Django run object or ID.
        tags: Space-separated tags. Will be used as file name if pandas objects
            are parsed.
        value: A file instance, a file path or a pandas.DataFrame/Series (will
            be converted to file via to_csv) or a dictionary of any of those.

        Returns: ResultFile (Django model) instance or list of instances.
        """
        errmsg = ('%s is not a file instance, existing path or ' % value +
                  'pandas DataFrame/Series or dictionary of those.')

        def is_valid(fu):
            return hasattr(fu, 'read') or (type(fu) is str and osp.exists(fu))

        def insert_file(**kwargs):
            return self.browser.insert('resultfile', **kwargs)

        if isinstance(value, pa.DataFrame) or isinstance(value, pa.Series):
            fi = StringIO()
            value.to_csv(fi)
            fn = '_'.join(tags.split())+'.csv'
            f = insert_file(run=run, tags=tags, file=fi, filename=fn)
        elif type(value) == dict:
            assert all([is_valid(v) for v in value.values()]), errmsg
            f = [insert_file(run=run, file=v, tags=tags+' '+str(k))
                 for k, v in value.items()]
        elif is_valid(value):
            f = insert_file(run=run, tags=tags, file=value)
        else:
            raise IOError(errmsg)
        return f

    def save_run(self, indicators={}, files={}, **run_fields):
        """
        Save the current SWIM input/output as a run in the browser database.

        Arguments:
        ----------
        indicators:
            Dictionary of indicator values or the functions argument passed
            to self.result_indicators (dict or list of method/attribute names).
        files:
            Dictionary of file values (valid file, path or dict of those) or
            functions argument passed to self.result_files (dict or list of
            method/attribute names).
        **run_fields:
            Set fields of the run browser table. Default fields: notes, tags

        Optional settings:
        ------------------
        resultindicator_functions:
            List or dictionary of method or attribute names that return an
            indicator (float) or dictionary of those.
        resultfile_functions:
            List or dictionary of method or attribute names that return any of
            file instance, a file path or a pandas.DataFrame/Series (will be
            converted to file via to_csv) or a dictionary of any of those.

        Returns: Run object (Django model object).
        """
        assert type(indicators) is dict, 'indicators must be a dictionary.'
        assert type(files) is dict, 'files must be a dictionary.'
        # config
        sty, nbyr = self.config_parameters('iyr', 'nbyr')
        run_fields.update({'start': dt.date(sty, 1, 1),
                           'end': dt.date(sty + nbyr - 1, 12, 31)})
        # add parameter changes
        run_fields['parameters'] = self.changed_parameters()
        # create run
        run = self.browser.insert('run', **run_fields)

        # add files and indicators
        for tbl, a in [('resultindicator', indicators), ('resultfile', files)]:
            save_function = getattr(self, 'save_' + tbl)
            # insert passed ones
            for k, v in a.items():
                save_function(run, k, v)
            # handle settings variable
            set_var = getattr(self, tbl + '_functions', False)
            if set_var:
                # ensure dictionary items
                items = (set_var.items() if type(set_var) == dict
                         else zip(set_var, set_var))
                for n, m in items:
                    value = self._attribute_or_function_result(m)
                    save_function(run, n, value)
        return run

    def _attribute_or_function_result(self, m):
        em = "%s is not a valid method or attribute name." % m
        assert self.settings.is_valid(m), em
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

        verbose: print changes.

        Returns: List of dictionaries with parameter browser attributes.
        """
        changed = []
        # create dicts with (pnam, stationID): value
        bsnp = self.basin_parameters()
        scp = self.subcatch_parameters().T.stack().to_dict()
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
    """
    mmproject = modelmanager.project.setup(projectdir, resourcedir)
    # swim specific customisation of resourcedir
    defaultsdir = osp.join(osp.dirname(__file__), 'resources')
    modelmanager.utils.copy_resources(defaultsdir, mmproject.resourcedir,
                                      overwrite=True)
    # FIXME rename templates with project name in filename
    for fp in ['cod', 'bsn']:
        ppn = modelmanager.utils.get_paths_pattern('input/*.' + fp, projectdir)
        tp = osp.join(mmproject.resourcedir, 'templates')
        os.rename(osp.join(tp, 'input/%s.txt' % fp), osp.join(tp, ppn[0]))
    # load as a swim project
    project = Project(projectdir)
    return project
