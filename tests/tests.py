#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `swimpy` package."""

import os
import os.path as osp
import subprocess
import shutil
import unittest


import swimpy
from swimpy.tests import test_project

SWIM_TEST_PROJECT = 'swim/project/'

if not os.path.exists(SWIM_TEST_PROJECT):
    raise IOError('The SWIM test project is not located at: %s'
                  % SWIM_TEST_PROJECT)


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


if __name__ == '__main__':
    unittest.main()
