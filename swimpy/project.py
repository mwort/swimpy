# -*- coding: utf-8 -*-

"""
The main project module.
"""
import os
import os.path as osp
import shutil
import datetime as dt
import subprocess
import tempfile
from numbers import Number
from decimal import Decimal

import pandas as pa

import modelmanager

from swimpy import utils


class Project(modelmanager.Project):

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

        # Reading
        subcatch_parameters(<param> or <stationID>) -> pa.Series
        subcatch_parameters(<list of param/stationID>) -> subset pa.DataFrame
        subcatch_parameters() -> pa.DataFrame of entire table

        # Writing
        # assign values or list to parameter column / stationID row (maybe new)
        subcatch_parameters(<param>=value, <stationID>=<list-like>)
        # set individual values
        subcatch_parameters(<param>={<stationID>: value, ...})
        # override entire table, DataFrame must have stationID index
        subcatch_parameters(<pa.DataFrame>)
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
                return bsn.loc[k]
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
        print 'Execution took %s hh:mm:ss' % delta
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
        self._is_method(functionname, fail=True)
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

    def result_indicators(self, functions=None):
        """
        Evaluate result indicators.

        functions:
            List of names of method or attribute names that return a result
            indicator (float) or dictionary of those. If None,
            'result_indicator_functions' setting is used.

        Returns: List of dictionaries containing indicator attributes.
        """
        functions = (functions if functions
                     else getattr(self, 'result_indicator_functions'))
        emsg = "Not all functions are method names: %r" % functions
        assert all([hasattr(self, m) for m in functions]), emsg
        indicators = []
        for i in functions:
            try:
                iv = (getattr(self, i)() if self._is_method(i)
                      else getattr(self, i))
            except Exception:
                raise Exception('Failed to evaluate indicator function %s' % i)
            emsg = ('Indicator function %s did not return a number ' % i +
                    'or a dictionary of numbers. Instead: %r' % iv)
            if isinstance(iv, Number):
                indicators += [dict(name=i, value=iv, tags=None)]
            elif type(iv) is dict:
                assert all([isinstance(v, Number) for v in iv.values()]), emsg
                indicators += [dict(name=i, value=v, tags=k)
                               for k, v in iv.items()]
            else:
                raise IOError(emsg)
        return indicators

    def result_files(self, functions=None):
        """
        Get all result files.

        functions:
            List of metho or attribute names that return a file instance, a
            file path or a pandas.DataFrame/Series (via to_csv). If None,
            'result_file_functions' setting is used.

        Returns: List of dictionaries containing attributes including the
                 'file' entry.
        """
        functions = (functions if functions
                     else getattr(self, 'result_file_functions'))
        emsg = ("Not all functions are method names: %r" % functions)
        assert all([hasattr(self, m) for m in functions]), emsg

        def is_valid(f):
            return isinstance(f, file) or (type(f) is str and osp.exists(f))
        files = []
        for f in functions:
            try:
                fv = (getattr(self, f)() if self._is_method(f)
                      else getattr(self, f))
            except Exception:
                raise Exception('Failed to call result file function %s' % f)
            if isinstance(fv, pa.DataFrame) or isinstance(fv, pa.Series):
                fi = tempfile.SpooledTemporaryFile()
                fv.to_csv(fi)
                files += [dict(file=fi, tags=f)]
            elif type(fv) == dict:
                assert all([is_valid(v) for v in fv.values()])
                files += [dict(file=v, tags='%s %s' % (f, k))
                          for k, v in fv.items()]
            elif is_valid(fv):
                files += [dict(file=fi, tags=f)]
            else:
                raise IOError('%s did not return a file instance, existing ' +
                              'path or pandas DataFrame/Series. Instead: %r'
                              % (f, fv))
        return files

    def save_run(self, indicators=None, files=None, **run_fields):
        """
        Save the current SWIM input/output as a run in the browser database.

        Arguments:
        ----------
        indicators:
            functions argument passed to self.result_indicators
        files:
            functions argument passed to self.resul_files
        **run_fields:
            Set fields of the run browser table. Default fields: notes, tags

        Returns: Run object (Django model object).
        """
        runattr = [('resultindicator', self.result_indicators(indicators)),
                   ('resultfile', self.result_files(files)),
                   ('parameter', self.changed_parameters())]
        # config
        sty, nbyr = self.config_parameters('iyr', 'nbyr')
        run_fields.update({'start': dt.date(sty, 1, 1),
                           'end': dt.date(sty + nbyr - 1, 12, 31)})
        run = self.browser.insert('run', **run_fields)
        for table, values in runattr:
            for attr in values:
                self.browser.insert(table, run=run, **attr)
        return run

    def _is_method(self, methodname, fail=False):
        ismethod = (hasattr(self, methodname) and
                    hasattr(getattr(self, methodname), '__call__'))
        if fail:
            assert ismethod, "%s is not a project method." % methodname
        return ismethod

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
        for k, v in bsnp.items() + scp.items():
            n, sid = k if type(k) == tuple else (k, None)
            # convert to minimal precision decimal via string
            dv = Decimal(str(v))
            saved = self.browser.get_table('parameter', name=n, tags=sid)
            if not saved or saved[-1]['value'] != dv:
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
