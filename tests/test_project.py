#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `swimpy` package."""

import os
import os.path as osp
import subprocess
import shutil
import unittest

import swimpy


SWIM_TEST_PROJECT = 'swim/project/'

if not os.path.exists(SWIM_TEST_PROJECT):
    raise IOError('The SWIM test project is not located at: %s'
                  % SWIM_TEST_PROJECT)


class ProjectTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.project = swimpy.project.setup(SWIM_TEST_PROJECT)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.project.resourcedir)


class TestSetup(unittest.TestCase):

    resourcedir = osp.join(SWIM_TEST_PROJECT, 'swimpy')

    def test_setup(self):
        self.project = swimpy.project.setup(SWIM_TEST_PROJECT)
        self.assertTrue(isinstance(self.project, swimpy.Project))
        self.assertTrue(osp.exists(self.resourcedir))
        self.assertTrue(osp.exists(self.project.resourcedir))
        for a in ['browser', 'clones', 'templates']:
            self.assertTrue(hasattr(self.project, a))
            self.assertIsNot(getattr(self.project, a), None)

    def test_setup_commandline(self):
        subprocess.call(['swimpy', 'setup',
                         '--projectdir=%s' % SWIM_TEST_PROJECT])
        self.assertTrue(osp.exists(self.resourcedir))

    def tearDown(self):
        shutil.rmtree(self.resourcedir)


class TestParameters(ProjectTestCase):

    def test_basin_parameters(self):
        bsn = self.project.basin_parameters()
        self.assertEqual(type(bsn), dict)
        self.assertGreater(len(bsn), 0)
        for k, v in bsn.items():
            self.assertEqual(self.project.basin_parameters(k), v)

    def test_config_parameters(self):
        cod = self.project.config_parameters()
        self.assertEqual(type(cod), dict)
        self.assertGreater(len(cod), 0)
        for k, v in cod.items():
            self.assertEqual(self.project.config_parameters(k), v)

    def test_subcatch_parameters(self):
        from pandas import DataFrame, Series
        # read
        sbc = self.project.subcatch_parameters()
        self.assertIsInstance(sbc, DataFrame)
        BLKS = self.project.subcatch_parameters('BLANKENSTEIN')
        self.assertIsInstance(BLKS, Series)
        roc2 = self.project.subcatch_parameters('roc2')
        self.assertIsInstance(roc2, Series)
        # write
        self.project.subcatch_parameters(roc2=1)
        self.assertEqual(self.project.subcatch_parameters('roc2').mean(), 1)
        self.project.subcatch_parameters(BLANKENSTEIN=2)
        BLKS = self.project.subcatch_parameters('BLANKENSTEIN').mean()
        self.assertEqual(BLKS, 2)
        newparamdict = {'roc2': 3.0, 'roc4': 10.0}
        HOF = self.project.subcatch_parameters('HOF')
        for k, v in newparamdict.items():
            HOF[k] = v
        self.project.subcatch_parameters(HOF=newparamdict)
        self.assertTrue((self.project.subcatch_parameters('HOF') == HOF).all())
        # write entire DataFrame
        self.project.subcatch_parameters(sbc.copy())
        nsbc = self.project.subcatch_parameters()
        self.assertTrue((nsbc == sbc).all().all())


if __name__ == '__main__':
    unittest.main()
