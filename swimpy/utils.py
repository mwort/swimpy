"""
Module for utility functionality unrelated to SWIM.
"""
from __future__ import print_function
import os
import os.path as osp
import subprocess
import warnings
import sys
import time
import datetime as dt
import io

import numpy as np
from modelmanager.settings import parse_settings

import swimpy


class cluster(object):
    """Simple plugin to abstract interaction with SLURM or another job manager.
    """
    plugin = ['__call__']

    class _job(object):
        """A dict-like store of slurm job attributes provided through sacct."""

        def __init__(self, id, **attributes):
            assert type(id) == int
            self.id = id
            self._keys = [v.lower() for v in self._sacct('-e').split()]
            self.__dict__.update(attributes)
            return

        def _sacct(self, *args):
            cmds = ["sacct", "-j", str(self.id)]+list(args)
            return subprocess.check_output(cmds).strip()

        def cancel(self):
            return subprocess.check_call(['scancel', str(self.id)])

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
            stdout = self._sacct('-Pn', '--format=%s' % key)
            # only return the first line to avoid duplicates from jobsteps
            return stdout.split("\n")[0].strip()

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
            rid = cluster._job(int(out.split()[-1]), resourcedir=cdir,
                               stderr=header['error'], stdout=header['output'],
                               jobfile=jcfpath, jobname=jobname, **slurmargs)
        else:
            rid = None
            print('Would execute: %s' % (' '.join(submit)))
        return rid

    def run_parallel(self, clones=None, args=None, timeout=None,
                     preprocess='basin_parameters', prefix='run_parallel',
                     **runkw):
        """Run SWIM in parallel using cluster jobs or multiprocessing.

        Arguments
        ---------
        clones : list | int
            List of clones with unique project names or max. number of clones
            to create if args is not None. If args is None, clones will
            only be run.
        args : list of dicts
            Arguments to parse to the preprocess function.
        timeout : dict | datetime.timedelta instance
            Limit job time and raise RuntimeError after timeout is elapsed.
            Parse any keyword as dict to datetime.timedelta, e.g. hours, days,
            minutes, seconds. Default: {'hours': 24}
        preprocess : str
            Name or dotted address of the project function to call with each
            entry of args.
        prefix : str
            A prefix for clone names (if they need to be created) and
            identification run tags (<prefix>_<pid>).
        runkw :
            Keywords to parse to the project.run method.

        Returns
        -------
        <django.db.QuerySet>
            Interable set of run instances.
        """
        st = dt.datetime.now()
        # check input
        lt = (list, tuple,)
        assert (type(clones) in lt+(int,)) or (type(args) in lt)
        if args:
            if clones is None:
                clones = len(args)
            assert all([type(i) == dict for i in args])
            assert type(preprocess) == str
            try:
                self.project.settings[preprocess]
            except AttributeError:
                raise AttributeError('%r is not a valid project function.'
                                     % preprocess)
        if type(clones) == int:
            assert args and preprocess
            clones = self.create_clones(clones, prefix=prefix)
        timeout = timeout or {'hours': 24}
        assert (type(timeout) == dict) or isinstance(timeout, dt.timedelta)
        timeout = dt.timedelta(**timeout) if type(timeout) == dict else timeout

        slurmargs = {'time': int(round(timeout.total_seconds()/60.))}
        tag = prefix + '_' + str(os.getpid())
        runkw.setdefault('cluster', {})
        deftag = runkw.setdefault('tags', '')
        queue = args or clones

        while len(queue) > 0:
            slurm_jobs = []
            mp_jobs = []
            n = min(len(queue), len(clones))
            qclones = clones[:n]
            # preprocess
            if args:
                for clone, a in zip(qclones, queue[:n]):
                    try:
                        clone.settings[preprocess](**a)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        m = '\nAn exception occurred while running %s.%s(**%r)'
                        raise RuntimeError(str(e) + m % (clone, preprocess, a))
            # run
            for clone in qclones:
                runkw['cluster'].update(dict(jobname=clone.clonename,
                                             slurmargs=slurmargs))
                runkw['tags'] = ' '.join([deftag, tag, clone.clonename])
                try:
                    job = clone.run(**runkw)
                    slurm_jobs.append(job)
                except OSError:
                    jf = osp.join(self.project.cluster.resourcedir,
                                  clone.clonename+'.py')
                    mp_jobs.append((jf, clone.projectdir))
            # remove run items from queue
            queue = queue[n:]
            # wait for runs to finish
            if mp_jobs:
                self.mp_process(mp_jobs)
            else:
                self.wait(slurm_jobs, timeout=timeout)

        runs = self.project.browser.runs.filter(
                tags__contains=tag, time__gt=st)
        return runs

    def create_clones(self, n, prefix='clone', **clonekw):
        """Create n clones named <prefix>_0-n.
        """
        cn = prefix + ('_%' + '0%0ii' % len(str(n - 1)))
        clones = [self.project.clone(cn % i, **clonekw) for i in range(n)]
        return clones

    def wait(self, jobs, timeout=None, interval=5):
        """Wait until all jobs are COMPLETED as per job.state.

        Arguments
        ---------
        jobs : list
            List of jobs to poll.
        interval : int seconds
            Polling interval in seconds.
        timeout : dict or datetime.timedelta instance
            Raise RuntimeError after timeout is elapsed. Parse any keyword as
            dict to datetime.timedelta, e.g. hours, days, minutes, seconds.
        """
        st = dt.datetime.now()
        # \u29D6 for hour glass removed
        ms = u"\r\033[KWaiting for %s runs (status: %s) for %s hh:mm:ss"
        ndone = 0
        njobs = len(jobs)
        status = {}
        while ndone < njobs:
            et = dt.datetime.now()-st
            if timeout and et > timeout:
                em = '%s runs not found within %s hh:mm:ss'
                raise RuntimeError(em % (njobs, timeout))
            ss = ['%s %s' % (n, s) for s, n in sorted(status.items())]
            sys.stdout.write(ms % (njobs-ndone, ', '.join(ss), et))
            sys.stdout.flush()
            time.sleep(interval)
            status = self.aggregated_job_status(jobs)
            if 'FAILED' in status or 'TIMEOUT' in status:
                self._raise_failed(jobs)
            ndone = status.get('COMPLETED', 0)
        # \u2713 for complete tick remove
        cmsg = u"\r\033[KCompleted %s runs in %s hh:mm:ss\n"
        sys.stdout.write(cmsg % (njobs, et))
        sys.stdout.flush()
        return

    def _raise_failed(self, jobs):
        failed = []
        for j in jobs:
            st = j.state
            if st == 'RUNNING':
                j.cancel()
            elif st == 'FAILED':
                with open(j.stderr) as f:
                    stderr = f.read()
                failed.append((j, stderr))
            elif st == 'TIMEOUT':
                failed.append((j, 'timed out.'))
        errors = '\n\n'.join([str(jn)+':\n'+se for jn, se in failed])
        nf = len(failed)
        raise RuntimeError('%i SLURM jobs failed/timedout:\n' % nf + errors)

    @staticmethod
    def aggregated_job_status(jobs):
        """Return the aggregated job status of a list of jobs in a dict."""
        status = {}
        for j in jobs:
            s = j.state
            status.setdefault(s, 0)
            status[s] += 1
        return status

    def mp_process(self, jobfileprojectdir):
        """Run the jobfiles in mp_jobs through multiprocessing.
        """
        import multiprocessing
        ncpu = min(len(jobfileprojectdir), multiprocessing.cpu_count())
        msg = 'Using multiprocessing on %s CPUs.' % ncpu
        warnings.warn(msg)
        pool = multiprocessing.Pool()
        pool.map(mp_process_clone, jobfileprojectdir)
        return


def mp_process_clone(clonejobfclonedir):
    """Execute a cluster jobfile with the same name as the clone in the clone's
    projectdir. This function needs to be here so that it can be pickled by the
    multiprocessing.Pool.
    """
    jf, cd = clonejobfclonedir
    os.chdir(os.path.join(cd))
    # create stderr, stdout files
    jpre = os.path.splitext(jf)[0]
    stderr, stdout = [open(jpre + '.%s' % s, 'w') for s in ('err', 'out')]
    # start subprocess
    try:
        ret = subprocess.check_call(['python', jf], stdout=stdout,
                                    stderr=stderr)
    except subprocess.CalledProcessError as cpe:
        raise Exception(str(cpe))
    stderr.close()
    stdout.close()
    return ret


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
        Aggregate to month or day-of-year mean regime. freq must be 'm' | 'd'.
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
    Path relative to settings.py file::

        import os.path as osp
        att_path = osp.join(osp.dirname(__file__), 'stations_attribute.csv')
        q_path = osp.join(osp.dirname(__file__), 'stations_q_data.csv')

    Read discharge data from file::

        _q = pd.read_csv(q_path, parse_dates=[0], index_col=0)

    Now read attributes from a csv file and add discharge. Make sure the index
    refers to the same IDs as your gauges.output file::

        stations = pd.read_csv(att_path, index_col=0)
        stations.daily_discharge_observed = _q

    Or from a GRASS table and csv Q data (requires GRASS settings)::

        import modelmanager.plugins.grass as mmgrass
        class stations(mmgrass.GrassAttributeTable):
            vector = 'stations_snapped'
            key = None  # specify here the column of your SWIM IDs
            daily_discharge_observed = _q

    """
    def __init__(self, project):
        u = (swimpy.__docs__ +
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


def upstream_ids(id, fromtoseries, maxcycle=1e6):
    """Return all ids upstream of id given a from (index) to (values) map.
    """
    s = [id]
    ids = []
    cycles = 0
    while len(s) > 0:
        si = []
        for i in s:
            si.extend(list(fromtoseries[fromtoseries == i].index))
        ids.extend(si)
        s = si
        cycles += 1
        if cycles > maxcycle:
            raise RuntimeError('maxcycles reached. Circular fromto?')
    return ids
