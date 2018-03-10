#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `swimpy` package."""

import os
import os.path as osp
import subprocess
import shutil
import unittest
import cProfile, pstats


import swimpy
from swimpy.tests import test_project

SWIM_TEST_PROJECT = 'project/'
SWIM_REPO_PROJECT = '../dependencies/swim/project'

if not os.path.exists(SWIM_TEST_PROJECT):
    shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT, symlinks=True)


class TestSetup(unittest.TestCase):

    resourcedir = osp.join(SWIM_TEST_PROJECT, 'swimpy')

    def test_setup(self):
        self.project = swimpy.project.setup(SWIM_TEST_PROJECT)
        self.assertTrue(isinstance(self.project, swimpy.Project))
        self.assertTrue(osp.exists(self.resourcedir))
        self.assertTrue(osp.exists(self.project.resourcedir))
        for a in ['browser', 'clone', 'templates']:
            self.assertTrue(hasattr(self.project, a))
            self.assertIsNot(getattr(self.project, a), None)

    def test_setup_commandline(self):
        subprocess.call(['swimpy', 'setup',
                         '--projectdir=%s' % SWIM_TEST_PROJECT])
        self.assertTrue(osp.exists(self.resourcedir))

    def tearDown(self):
        shutil.rmtree(self.resourcedir)


class ProjectTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.project = swimpy.project.setup(SWIM_TEST_PROJECT)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.project.resourcedir)


class TestParameters(ProjectTestCase, test_project.Parameters):
    pass


class TestProcessing(ProjectTestCase, test_project.Processing):
    pass


class TestRun(ProjectTestCase, test_project.Run):
    pass


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
