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
import codecs

import numpy as np
import pandas as pd
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
            return subprocess.check_output(cmds).decode().strip()

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
            # special treatment since slurmargs is often parsed by settings
            slurmargs.update(jobname.get('slurmargs', slurmargs))
            jobname = jobname['jobname']
        assert type(functionname) == str or type(script) == str
        if functionname:
            assert callable(self.project.settings[functionname])
            script = ("import swimpy\np=swimpy.Project()\np.%s(**%r)" %
                      (functionname, funcargs))
        # submit to slurm
        rid = self.submit_job(jobname, script, self.resourcedir, dryrun=dryrun,
                              chdir=self.project.projectdir, **slurmargs)
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
                * chdir: change to working directory
                * account: CPU accounting

        Example
        -------
        >>> submit_job('testjob', 'import swimpy; swimpy.Project().run()',
        ...            chdir='project/', dryrun=True)  # doctest: +ELLIPSIS
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

    @parse_settings
    def run_parallel(self, clones=None, args=None, time=None,
                     preprocess='config_parameters', prefix='run_parallel',
                     parallelism='jobs', mpi_master=False, mp_max_cpus=None, **runkw):
        """Run SWIM in parallel using cluster jobs or multiprocessing.

        Arguments
        ---------
        clones : list | int
            List of clones with unique project names or max. number of clones
            to create if args is not None. If args is None, clones will
            only be run.
        args : list of dicts
            Arguments to parse to the preprocess function.
        time : str | int, optional
            Slurm job time limit to reduce queuing times.
            Format is (see slurm manual): 'mins' | 'hh:mm'
            Default is the default of the QOS class.
        preprocess : str
            Name or dotted address of the project function to call with each
            entry of args.
        prefix : str
            A prefix for clone names (if they need to be created) and
            identification run tags (<prefix>_<pid>).
        parallelism : 'jobs' | 'mp' | 'mpi'
            Cluster processing method: submit as jobs or run on all available
            CPUs via shared-memory multiprocessing (mp) or via MPI.
        mpi_master : bool
            If using MPI, reserve process 0 as master without running SWIM to
            preserve its memory.
        runkw :
            Keywords to parse to the project.run method.

        Returns
        -------
        <django.db.QuerySet>
            Interable set of run instances.
        """
        st = dt.datetime.now()
        # check input
        clones, preprocess, args = self._check_args(clones, preprocess, args)
        runmethod = getattr(self, '_run_'+parallelism, None)
        if runmethod is None:
            raise RuntimeError('Cant find method %s' % parallelism)
        tag = prefix + '_' + str(os.getpid())
        deftag = runkw.setdefault('tags', '')

        if parallelism == 'jobs':
            runkw.setdefault('cluster', {})
            if time:
                runkw['cluster'].setdefault('slurmargs', {})
                runkw['cluster']['slurmargs'] = {'time': str(time)}

        if parallelism == 'mpi':
            comm = self._mpi_comm()
            rank, size = comm.Get_rank(), comm.Get_size()
            tag = comm.bcast(tag, root=0)
            if type(clones) == int and clones > size:
                clones = size
            self.mpi_master = mpi_master
        else:
            rank = 0

        if parallelism== 'mp' and mp_max_cpus:
           runkw['max_cpus'] = mp_max_cpus

        # create or convert clones to names
        if type(clones) == int:
            assert args and preprocess
            no = (rank - int(mpi_master)) if parallelism == 'mpi' else None
            clones_names = self._create_clones(clones, prefix=prefix, nonly=no)
        else:
            clones_names = [getattr(c, 'clonename', c) for c in clones]

        queue = args or clones_names
        while len(queue) > 0:
            n = min(len(queue), len(clones_names))
            qclones = clones_names[:n]
            # submit
            ag = queue[:n] if args else None
            runmethod(qclones, deftag+' '+tag, preprocess, ag, **runkw)
            # remove run items from queue
            queue = queue[n:]

        runs = self.project.browser.runs.filter(
                tags__contains=tag, time__gt=st)
        return runs

    def _run_jobs(self, clones, tag, preprocess, args, **runkw):
        """Run all clones by submitting them as jobs."""
        runkw.setdefault('cluster', {})
        slurm_jobs = []
        for clonen, a in zip(clones, args or [None]*len(clones)):
            clone = self.project.clone[clonen]
            if args:
                self._call(clone, preprocess, a)
                runkw['notes'] = str(a)
            runkw['cluster']['jobname'] = clone.clonename
            runkw['tags'] = ' '.join([tag, clone.clonename])
            job = self._call(clone, 'run', runkw)
            slurm_jobs.append(job)
        # wait for runs to finish
        self.wait(slurm_jobs)
        return

    def _run_mp(self, clones, tag, preprocess, args, max_cpus=None,**runkw):
        """Run the clones through multiprocessing."""
        import multiprocessing
        ncpu = min(len(clones), max_cpus or multiprocessing.cpu_count())
        msg = 'Using multiprocessing on %s CPUs.' % ncpu
        warnings.warn(msg)
        mp_jobs = []
        # create inputs to mp_process_clone
        for clonen, a in zip(clones, args or [None]*len(clones)):
            if args:
                self._call(clonen, preprocess, a)
                runkw['notes'] = str(a)
            runkw['tags'] = ' '.join([tag, clonen])
            runkw['quiet'] = osp.join(self.resourcedir, clonen+'.out')
            cpath = osp.join(self.project.clone_dir, clonen)
            mp_jobs.append((cpath, runkw.copy()))
        # wait for runs to finish
        pool = multiprocessing.Pool()
        pool.map(_mp_process_clone, mp_jobs)
        pool.close()
        return

    def _run_mpi(self, clones, tag, preprocess, args, **runkw):
        """Run clones using mpi4py."""
        mpim = int(self.mpi_master)
        nc = len(clones)
        comm = self._mpi_comm()
        rank, size = comm.Get_rank(), comm.Get_size()
        if rank == 0:
            msg = 'Not enough CPUs (%s) for %s clones.'
            assert nc <= size-mpim, msg % (size, nc)
            if nc < size-mpim:
                warnings.warn('Lower clones count than available CPUs. %s < %s'
                              % (nc, size-mpim))
        # if unneeded rank or master rank 0, wait until others have finished
        if rank-mpim >= nc or (mpim and rank == 0):
            comm.Barrier()
            return

        clone = self.project.clone[clones[rank-mpim]]
        if args:
            print('MPI preprocess %i/%i (rank/clones).' % (rank, nc))
            self._call(clone, preprocess, args[rank-mpim])
            runkw['notes'] = str(args[rank-mpim])
        runkw.pop('cluster', None)
        runkw['tags'] = ' '.join([tag, clone.clonename])
        # let rank 0 print to standard out, others to file in swimpy/cluster
        if rank > 0:
            runkw['quiet'] = osp.join(self.resourcedir, clone.clonename+'.out')
        print('MPI %i/%i running.' % (rank, nc))
        self._call(clone, 'run', runkw)
        print('MPI %i done.' % rank)
        # wait for all clones to finish before returning
        comm.Barrier()
        return

    def _mpi_comm(self):
        try:
            from mpi4py import MPI
        except ImportError:
            raise ImportError('mpi4py needed to run with mpi.')
        return MPI.COMM_WORLD

    def _check_args(self, clones, preprocess, args):
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
        else:
            preprocess = None
        return clones, preprocess, args

    def _create_clones(self, n, prefix='clone', nonly=None, **clonekw):
        """Create n clones named <prefix>_0-n.
        If nonly (int) is given, only this id's clone is created.
        Returns a list of clone names.
        """
        cn = prefix + ('_%' + '0%0ii' % len(str(n - 1)))
        clones = []
        for i in range(n):
            n = cn % i
            if nonly is None or i == nonly:
                self.project.clone(n, **clonekw)
            clones.append(n)
        return clones

    def _call(self, clone, functionpath, args):
        """Call function on clone with args."""
        clone = self.project.clone[clone] if type(clone) == str else clone
        try:
            r = clone.settings[functionpath](**args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            m = '\nAn exception occurred while running %s.%s(**%r)'
            errmsg = str(e)+m % (clone.clonename, functionpath, args)
            raise RuntimeError(errmsg)
        return r

    @parse_settings
    def wait(self, jobs, timeout=None, interval=5, fail=False):
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
        fail: bool
            Cancel all jobs if one has failed or timed out and raise
            RuntimeError.
        """
        st = dt.datetime.now()
        # \u29D6 for hour glass removed
        ms = u"\r\033[KWaiting for %s runs (status: %s) for %s hh:mm:ss"
        failed_status = ['FAILED', 'TIMEOUT']
        ndone = 0
        njobs = len(jobs)
        status = {}
        et = 0
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
            nfailed = 0
            failedst = set(failed_status).intersection(status)
            if failedst:
                nfailed = (sum([status.get(s) for s in failedst])
                           if not fail else self._raise_failed(jobs))
            ndone = status.get('COMPLETED', 0) + nfailed
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


def _mp_process_clone(clonedirkw):
    """Simple run function to use with multiprocessing.Pool.map.
    This function needs to be here so that it can be pickled by the
    multiprocessing.Pool.
    """
    clonedir, kw = clonedirkw
    clone = swimpy.Project(clonedir)
    clone.run(**kw)
    clone.browser.settings.unset()
    return


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

    It should be a pandas.DataFrame or subclass thereof of station information
    indexed by the same IDs used in `input/gauges.output`. It is possible to
    link it to the GRASS stations vector table (see examples). Observed
    discharge should be in `stations.daily_discharge_observed`.

    If a file called `daily_discharge_observed.csv` exists in the SWIMpy
    resource directory and has the below format, the
    `stations.daily_discharge_observed` attribute will be loaded::

        yyyy-mm-dd, station1, station2, ...
        2000-01-01,    100.0,    200.0, ...
        2000-01-02,    102.0,    201.0, ...
        2000-01-03,    103.0,    204.0, ...
        ...

    Examples
    --------
    Here are some examples of how to set this attribute in the `settings.py` of
    your project. Note that the settings file is loaded from various locations,
    i.e. dont rely on relative paths. It's best to use dynamic or absolute
    paths. For example, the path to the swimpy resource directory can be
    obtained like this::

        import os.path as osp
        _here = osp.dirname(__file__)

    Assuming you have your station information in `stations_info.csv` (one
    station per line, one attribute per column) and observed discharge in
    `q_data.csv` (one day per line with dates as YYYY-MM-DD in first column,
    one station per column) in well-formatted CSV files in the swimpy resource
    directory, a basic configuration would be::

        import pandas as pd
        stations = pd.read_csv(osp.join(_here, 'stations_info.csv'), index_col=0)
        _qdat = pd.read_csv(osp.join(_here, 'q_data.csv'), parse_dates=[0],
                            index_col=0)
        stations.daily_discharge_observed = _qdat

    Make sure the station info row index and discharge data column names refer
    to the same IDs as your gauges.output file. Refer to the `pandas.read_csv
    help page <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv>`_
    for help with reading data into a pandas DataFrame.

    To load the station infos from GRASS stations_snapped vector table
    (requires GRASS settings)::

        import modelmanager.plugins.grass as mmgrass
        class stations(mmgrass.GrassAttributeTable):
            vector = 'stations_snapped'
            key = None  # specify here the column of your SWIM IDs
            daily_discharge_observed = _qdat

    """

    #: Default observed discharge file in SWIMpy resource directory
    daily_discharge_observed_file = 'discharge.csv'

    def __init__(self, project):
        u = (swimpy.__docs__ +
             "/modules/utils.html#swimpy.utils.StationsUnconfigured")
        self.error = RuntimeError('The Stations attribute is unconfigured, '
                                  'for help see:\n' + u)
        # add daily_discharge_observed_file if it exists
        daily_discharge_observed_path = osp.join(
            project.resourcedir, self.daily_discharge_observed_file)
        if osp.exists(daily_discharge_observed_path):
            self.daily_discharge_observed = pd.read_csv(
                daily_discharge_observed_path, index_col=0, parse_dates=[0],
                date_parser=pd.Period)
        return

    def __getattr__(self, a):
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


class GRDCStation(pd.DataFrame):
    """A dataframe created from a GRDC discharge data file."""
    # some assumptions about GRDC files as class attributes
    fileEncoding = 'latin_1'
    dataStart = '# DATA'

    # avoid pandas complaining about new properties
    _metadata = ['path', 'header', 'header_attributes']

    def __init__(self, path):
        super(GRDCStation, self).__init__()
        # precess file
        with codecs.open(path, 'r', encoding=self.fileEncoding) as f:
            self.read_header(f)
            # read data and initialise DF with it
            super(GRDCStation, self).__init__(self.read(f))
        # file name
        self.path = path
        return

    def read_header(self, fobj):
        self.header = ''
        self.header_attributes = []
        for l in fobj:
            if l.startswith(self.dataStart):
                break
            elif l.startswith('#'):
                self.header += l
                if len(l.split(':')) == 2:
                    k, v = l.split(':')
                    clek = [''.join([c.lower() for c in string if c.isalnum()])
                            for string in k.split()]
                    k = '_'.join([c for c in clek if len(c) > 0])
                    self.__dict__.update({k: v.strip()})
                    self.header_attributes += [k]
        return

    def read(self, fobj):
        df = pd.read_csv(fobj, sep=str(self.field_delimiter),
                         index_col=0,
                         # faster then doing it in a loop afterw
                         parse_dates=[0],
                         engine='python')  # because of already open file
        # set -999 to na
        df[df == -999] = np.nan
        # day or month index
        if hasattr(df.index, 'to_period'):
            df.index = df.index.to_period()
        elif str(df.index[0]).endswith('00'):  # monthly
            df.index = pd.PeriodIndex(
                [i[:7] for i in df.index.astype(str)], freq='m')
        # format columns
        df.columns = [c.strip().lower() for c in df.columns]
        if 'hh:mm' in df.columns:
            df.drop('hh:mm', axis=1, inplace=True)
        return df

    def __repr__(self):
        dfrep = super(GRDCStation, self).__repr__().split(u'\n')
        header = self.header.split(u'\n')
        rep = dfrep[0] + '\n' + u'\n'.join(header + dfrep[1:])
        return rep.encode('utf8', 'ignore').decode()


def read_csv_multicol(path, indexcol='time', ifreq='d',
                      spacecol=None, **kwargs):
    """Read csv files of structure <time>, <spatial>, <variable(s)>.

    Output is a pd.DataFrame with DatetimeIndex and MultiIndex column
    ['variable', <spacecol>].
    
    Arguments
    ---------
    path: str
        File to read.
    indexcol: str
        Index name (treated as date index).
    ifreq: str
        Index frequency. Passed to pd.index.to_period().
    spacecol: str
        Name of column that is a spatial identifier (other columns are treated
        as 'variable' in the final MultiIndex column).
    **kwargs :
        Keywords to pandas.read_csv.
    """
    na_values = ['NA', 'NaN', -999, -999.9, -9999]
    df = pd.read_csv(path, skipinitialspace=True,
                    index_col=indexcol, parse_dates=True,
                    na_values=na_values, **kwargs)
    df.index = df.index.to_period(freq=ifreq)
    # multi-index columns
    df = pd.pivot_table(df, index='time', columns=[spacecol])
    df.columns.names = ['variable', spacecol]
    return df


def write_csv_multicol(df, path, spacecol=None, **kwargs):
    """Write pd.DataFrame with DatetimeIndex and MultiIndex column
    ['variable', <spacecol>] (e.g. read with read_csv_multicol()) to a csv file
    of structure <time>, <spatial>, <variable(s)>.
    
    Arguments
    ---------
    df: pd.DataFrame
        pd.DataFrame object with DatetimeIndex and MultiIndex column
        ['variable', <spacecol>].
    path: str
        File to write to.
    spacecol: str
        Name of column that is a spatial identifier (other columns are treated
        as 'variable' in the final MultiIndex column).
    **kwargs :
        Keywords to pandas.to_csv.
    """
    df_stack = df.stack()
    df_out = df_stack.reset_index(level=[spacecol])
    df_out.to_csv(path, index = True, na_rep='-9999', **kwargs)
    return
