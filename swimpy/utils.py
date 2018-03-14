"""
Module for utility functionality unrelated to SWIM.
"""
import os
import os.path as osp

import pandas as pd


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
    subprocess.call(['chmod', '+x', jcfpath])
    # submit
    submit = ['sbatch', jcfpath]
    if not dryrun:
        subprocess.call(submit)
    else:
        print('Would execute: %s' % (' '.join(submit)))
    return


class ProjectOrRunData(pd.DataFrame):
    """
    A representation of data read from either the SWIM project or Run instance.
    """
    swim_path = None
    plugin_functions = []

    def __init__(self, projectorrun):
        from swimpy.project import Project  # avoid circular import
        # init DataFrame
        pd.DataFrame.__init__(self)
        self.name = self.__class__.__name__
        # instantiated with project
        if isinstance(projectorrun, Project):
            self.project = projectorrun
            self.path = osp.join(self.project.projectdir, self.swim_path)
            self.read = self.from_project
        # instantiated with run
        elif hasattr(projectorrun, 'resultfiles'):
            self.run = projectorrun
            self.read = self.from_run
        else:
            raise IOError('Run includes no saved files.')
        # read file
        pd.DataFrame.__init__(self, self.read())
        return

    def from_run(self):
        """
        Read data from a run instance with resultfiles.
        """
        # find file
        fileqs = (self.run.resultfiles.filter(tags__contains=self.name) or
                  self.run.resultfiles.filter(file__contains=self.name))
        if fileqs.count() > 1:
            print('Found two resultfiles for %s, using last!' % self.name)
        elif fileqs.count() == 0:
            raise IOError('No resultfile found for %s!' % self.name)
        fileobj = fileqs.last()
        # determine function according to extension
        ext = osp.splitext(fileobj.file.path)[1][1:]  # no dot
        readmethodname = 'from_' + ext
        if not hasattr(self, readmethodname):
            raise NotImplementedError('No method %s to read file %s defined!' %
                                      (readmethodname, fileobj.file.path))
        self.path = fileobj.file.path
        return getattr(self, readmethodname)()

    def from_project(self):
        """!Overwrite me!"""
        raise NotImplementedError('Cant read this ProjectOrRunData from '
                                  'project, define a from_project method!')
