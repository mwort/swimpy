"""
Module for utility functionality unrelated to SWIM.
"""
from __future__ import print_function
import os
import os.path as osp
import subprocess

import numpy as np
from modelmanager.settings import parse_settings


class cluster(object):
    """Simple plugin to abstract interaction with SLURM or another job manager.
    """
    plugin = ['__call__']

    class _job(object):
        """A dict-like store of slurm job attributes provided through sacct."""
        def __init__(self, id):
            assert type(id) == int
            self.id = id
            self._keys = [v.lower() for v in self._sacct('-').split()]
            return

        def _sacct(self, *args):
            return subprocess.check_output(["sacct", "-j", str(self.id)]+args)

        def status(self, _print=True):
            ks, vs = self._sacct("-lP").split('\n')
            dict = {k.lower(): v for k, v in zip(ks.split('|'), vs.split('|'))}
            if _print:
                [print('%s: %s' % s) for s in dict.items()]
            else:
                return dict

        def keys(self):
            return self._keys

        def __getattr__(self, key):
            key = key.lower()
            assert key in self.keys(), key + ' not a valid job attribute.'
            return self._sacct('-Pn', '--format=%s' % key)

        def __getitem__(self, key):
            assert type(key) == str
            return self.__getattr__(key)

        def __repr__(self):
            p = (self.id, self.state)
            return "<swimpy.utils.cluster._job ID=%i %s>" % p

    def __init__(self, project):
        self.project = project
        # dir for slurm job, output, error files
        self.resourcedir = osp.join(project.resourcedir, 'cluster')
        if not osp.exists(self.resourcedir):
            os.mkdir(self.resourcedir)
        return

    @parse_settings
    def __call__(self, jobname, functionname=None, script=None, dryrun=False,
                 slurmargs={}, **funcargs):
        """
        Run a project function (method) by submitting it to SLURM.

        Arguments
        ---------
        jobname : str | dict
            SLURM job name. As a convenience argument, a dict may be parsed to
            set the other arguments.
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
        if type(jobname) == dict:
            assert 'jobname' in jobname, 'No jobname given in %s' % jobname
            functionname = jobname.get('functionname', functionname)
            script = jobname.get('script', script)
            dryrun = jobname.get('dryrun', dryrun)
            slurmargs = jobname.get('slurmargs', slurmargs)
            jobname = jobname['jobname']
        assert type(functionname) == str or type(script) == str
        if functionname:
            assert callable(self.project.settings[functionname])
            script = ("import swimpy\np=swimpy.Project()\np.%s(**%r)" %
                      (functionname, funcargs))
        # submit to slurm
        rid = self.submit_job(jobname, script, self.resourcedir, dryrun=dryrun,
                              workdir=self.project.projectdir, **slurmargs)
        return rid

    @staticmethod
    def submit_job(jobname, scriptstr, outputdir='.', dryrun=False,
                   **slurmargs):
        '''
        Submit the script string as a python script to slurm.

        Arguments
        ---------
        jobname : str
            Job identifier without spaces.
        scriptstr : str
             Valid python code string (ensure correct indent and linebreaks).
        outputdir : str path
            Directory where the script, error and output files are written.
        dryrun : bool
            If true, dont submit job but just write jobfile.
        **slurmkwargs
            Any additional slurm header arguments, some useful ones:
                * qos: job class (short, medium, long)
                * workdir: working directory
                * account: CPU accounting

        Example
        -------
        >>> submit_job('testjob', 'import swimpy; swimpy.Project().run()',
        ...            workdir='project/', dryrun=True)  # doctest: +ELLIPSIS
        Would execute: sbatch .../testjob.py
        '''
        import ast
        import subprocess
        try:
            ast.parse(scriptstr)
        except SyntaxError:
            raise SyntaxError('scriptstr is not valid python code. %s'
                              % scriptstr)
        # defaults
        cdir = osp.abspath(outputdir)
        header = {'job-name': jobname,
                  'error': os.path.join(cdir, '%s.err' % jobname),
                  'output': os.path.join(cdir, '%s.out' % jobname),
                  }
        header.update(slurmargs)

        # setup jobfile
        jcfpath = os.path.join(cdir, jobname + '.py')
        jcf = open(jcfpath, 'w')
        jcf.write('#!/usr/bin/env python \n')
        # SLURM lines
        for c, v in header.items():
            jcf.write('#SBATCH --%s=%s\n' % (c, v))
        jcf.write(scriptstr)
        jcf.close()
        # make file executable
        subprocess.call(['chmod', '+x', jcfpath])
        # submit
        submit = ['sbatch', jcfpath]
        if not dryrun:
            out = subprocess.check_output(submit)
            print(out, end='')
            rid = cluster._job(int(out.split()[-1]))
        else:
            rid = None
            print('Would execute: %s' % (' '.join(submit)))
        return rid


def aggregate_time(obj, freq='d', regime=False, resample_method='mean',
                   regime_method='mean'):
    """Resample a DataFrame or Series to a different frequency and/or regime.

    Arguments
    ---------
    obj : pd.Series | pd.DataFrame
        Must have a time-like index.
    freq : <pandas frequency>
        Aggregate to different frequency, any pandas frequency string
        or object is allowed.
    regime : bool
        Aggregate to month or day-of-year mean regime. freq must be 'a' | 'd'.
    resample_method :
        The aggregator for the resample method. See DataFrame.groupby.agg.
    regime_method :
        The aggregator for the regime groupby.agg. See DataFrame.groupby.agg.
    """
    assert hasattr(obj, 'index') and hasattr(obj.index, 'freq')
    if freq != obj.index.freq:
        obj = obj.resample(freq).aggregate(resample_method)
    if regime:
        if freq == 'd':
            igb = obj.index.dayofyear
        elif freq == 'm':
            igb = obj.index.month
        else:
            raise TypeError("freq must be either 'm' or 'd' with "
                            "regime=True.")
        obj = obj.groupby(igb).agg(regime_method)
    return obj


class StationsUnconfigured(object):
    """Dummy stations plugin. The stations setting needs to be configured.
    It should be a pandas.DataFrame or subclass thereof, e.g.
    modelmanager.plugins.grass.GrassAttributeTable to link it to a GRASS table.

    Examples
    --------
    From a csv file with attributes and another one with Q data::

        stations = pd.read_csv('stations_attribute.csv', index_col=0)
        _q = pd.read_csv('stations_q_data.csv', parse_dates=[0], index_col=0)
        stations['daily_discharge_observed'] = dict(_q.iteritems())

    From a GRASS table and csv Q data (requires GRASS settings)::

        import modelmanager.plugins.grass as mmgrass
        _q = pd.read_csv('stations_q_data.csv', parse_dates=[0], index_col=0)
        class stations(mmgrass.GrassAttributeTable):
            vector = 'stations_snapped',
            add_attributes = {'daily_discharge_observed': _q.to_dict('series')}

    """
    def __init__(self, project):
        u = (project.doc_url if hasattr(project, 'doc_url') else '<doc-url>' +
             "/modules/utils.html#swimpy.utils.StationsUnconfigured")
        self.error = RuntimeError('The Stations attribute is unconfigured, '
                                  'for help see:\n' + u)
        return

    def __getattr__(self, a):
        raise self.error

    def __repr__(self):
        raise self.error

    def __getitem__(self, k):
        raise self.error
