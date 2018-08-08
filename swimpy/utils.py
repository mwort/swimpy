"""
Module for utility functionality unrelated to SWIM.
"""
from __future__ import print_function
import os
import os.path as osp


def slurm_submit(jobname, scriptstr, outputdir='.', dryrun=False, **slurmargs):
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
    >>> slurm_submit('testjob', 'import swimpy; swimpy.Project().run()',
    ...              workdir='project/', dryrun=True)  # doctest: +ELLIPSIS
    Would execute: sbatch .../testjob.py
    '''
    import ast
    import subprocess
    try:
        ast.parse(scriptstr)
    except SyntaxError:
        raise SyntaxError('scriptstr is not valid python code. %s' % scriptstr)
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
        out = subprocess.call(submit)
        print(out, end='')
        rid = int(out.split()[-1])
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
