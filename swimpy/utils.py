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

    def from_run(self, **readkwargs):
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
        self.path = fileobj.file.path
        return self.reader_by_ext(fileobj.file.path)(**readkwargs)

    def from_project(self, **kw):
        """!Overwrite me!"""
        raise NotImplementedError('Cant read this ProjectOrRunData from '
                                  'project, define a from_project method!')

    def from_gzip(self, **readkwargs):
        readkwargs['compression'] = 'gzip'
        return self.reader_by_ext(osp.splitext(self.path)[0])(**readkwargs)

    def reader_by_ext(self, path):
        """
        Return the read method from_* using the self.path extension.
        Raises a NotImplementedError if none found.
        """
        ext = osp.splitext(path)[1][1:]  # no dot
        readmethodname = 'from_' + ext
        if not hasattr(self, readmethodname):
            raise NotImplementedError('No method %s to read file %s defined!' %
                                      (readmethodname, path))
        return getattr(self, readmethodname)


class ReadWriteDataFrame(pd.DataFrame):
    """
    A representation of data read and written to file.

    Intended for use as a superclass of a @propertyplugin to map file table to
    a pandas DataFrame.

    Usage:
    ------
    @propertyplugin
    class ProjectData(ReadWriteData):
        path = 'some/relative/path.csv'

        def read(self, *kw):
            <read data from file and assign pd.DataFrame.__init__(self, data)>
        def write(self, *kw):
            <write data from file, possible error/consistency checking>

    """
    path = None
    plugin_functions = []

    def __init__(self, project):
        # init DataFrame
        pd.DataFrame.__init__(self)
        self.name = self.__class__.__name__
        self.project = project
        self.path = osp.join(self.project.projectdir, self.path)
        errmsg = self.name + 'file does not exist: ' + self.path
        assert osp.exists(self.path), errmsg
        # read file
        self.read()
        return

    def __call__(self, data=None, **setvalues):
        """
        Assign read data from file and optionally set and write new values.

        data: <2D-array-like>
            Set entire dataframe.
        **setvalues: <array-like> | <dict>
            Set columns or rows by kew. Subset of values can be set by parsing
            a dict. Creates new row if key is neither in columns or index.
        """
        if data is not None:
            pd.DataFrame.__init__(self, data)
            self.write()
        elif setvalues:
            self.read()
            for k, v in setvalues.items():
                ix = slice(None)
                if type(v) == dict:
                    ix, v = zip(*v.items())
                if k in self.columns:
                    self.loc[ix, k] = v
                else:
                    self.loc[k, ix] = v
            self.write()
        else:
            self.read()
        return

    def __repr__(self):
        rpr = '<%s: %s >\n' % (self.name, osp.relpath(self.path))
        return rpr + pd.DataFrame.__repr__(self)

    def read(self, **kwargs):
        """
        Override me and reinitialise the data by calling:
            pd.DataFrame.__init__(self, data)
        """
        raise NotImplementedError('Reading of %s not implemented.' % self.name)

    def write(self, **kwargs):
        """
        Override me. Error checking and writing to file should be done here.
        """
        raise NotImplementedError('Writing of %s not implemented.' % self.name)
