"""Tests for swimpy project setup (standalone, no project fixture needed)."""
import os.path as osp
import subprocess
import shutil

import pytest
import swimpy

from conftest import (SWIM_REPO_PROJECT, ProjectTestCase, SWIM_TEST_PROJECT, SWIM_REPO,
                      _rmtree_nfs)


class TestSetup:

    projectdir = SWIM_TEST_PROJECT
    resourcedir = osp.join(projectdir, 'swimpy-project')

    def test_setup(self):
        shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT, dirs_exist_ok=True)
        project = swimpy.project.setup(self.projectdir, name='test',
                                       gitrepo=SWIM_REPO)
        assert isinstance(project, swimpy.Project)
        assert osp.exists(self.resourcedir)
        assert osp.exists(project.resourcedir)
        assert osp.exists(osp.join(self.projectdir, 'input'))
        assert osp.exists(osp.join(self.projectdir, 'output'))
        assert osp.exists(osp.join(self.projectdir, 'blankenstein_parameters.nml'))
        assert osp.exists(osp.join(self.projectdir, 'swim'))
        for a in ['browser', 'clone', 'templates']:
            assert hasattr(project, a)
            assert getattr(project, a) is not None
        project.browser.settings.unset()
        _rmtree_nfs(project.projectdir)

    def test_setup_commandline(self):
        shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT, dirs_exist_ok=True)
        subprocess.call(['swimpy', 'setup', '--name=test',
                         '--projectdir=' + self.projectdir,
                         '--gitrepo=' + SWIM_REPO])
        assert osp.exists(self.resourcedir)
        project = swimpy.Project(self.projectdir)
        project.browser.settings.unset()
        _rmtree_nfs(project.projectdir)
