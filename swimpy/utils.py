"""
Module for utility functionality unrelated to SWIM.
"""
import os
import os.path as osp


def slurm_submit(jobname, scriptstr, outputdir='.', dryrun=False, **slurmargs):
    '''
    Submit the script string as a python script to slurm.

    jobname: job identifier
    scriptstr: valid python code string (ensure correct indent and linebreaks)
    outputdir: directory where the script, error and output files are written
    slurmkwargs: any additional slurm header arguments, some useful ones:
        qos: job class (short, medium, long)
        workdir: working directory
        account: CPU accounting
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
    subprocess.call(['chmod', '740', jcfpath])
    # submit
    submit = ['sbatch', jcfpath]
    if not dryrun:
        subprocess.call(submit)
    else:
        print('Would execute: %s' % (' '.join(submit)))
    return
