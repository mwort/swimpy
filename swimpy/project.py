# -*- coding: utf-8 -*-

"""
The main project module.
"""
import os
import os.path as osp
import shutil

import modelmanager


class Project(modelmanager.Project):
    pass


def setup(projectdir='.', resourcedir='swimpy'):
    """
    Setup a swimpy project.
    """
    mmproject = modelmanager.project.setup(projectdir, resourcedir)
    # swim specific customisation of resourcedir
    defaultresourcedir = osp.join(osp.dirname(__file__), 'resources')
    for path, dirs, files in os.walk(defaultresourcedir, topdown=True):
        relpath = os.path.relpath(path, defaultresourcedir)
        for f in files:
            dst = osp.join(mmproject.resourcedir, relpath, f)
            shutil.copy(osp.join(path, f), dst)
        for d in dirs:
            dst = osp.join(mmproject.resourcedir, relpath, d)
            if not osp.exists(dst):
                os.mkdir(path)
    # load as a swim project
    project = Project(projectdir)
    return project
