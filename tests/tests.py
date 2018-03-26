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

TEST_GRASSDB = 'grassdb'
MSWIM_GRASSDB = '../dependencies/m.swim/test/grassdb'

if not os.path.exists(SWIM_TEST_PROJECT):
    shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT, symlinks=True)
if not os.path.exists(TEST_GRASSDB):
    shutil.copytree(MSWIM_GRASSDB, TEST_GRASSDB, symlinks=True)


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
        self.project.browser.settings.unset()
        shutil.rmtree(self.project.resourcedir)


class TestParameters(ProjectTestCase, test_project.Parameters):
    pass


class TestProcessing(ProjectTestCase, test_project.Processing):
    pass


class TestRun(ProjectTestCase, test_project.Run):
    pass


class TestGrass(ProjectTestCase):
    grass_settings = dict(
        grass_db = TEST_GRASSDB,
        grass_location = "utm32n",
        grass_mapset =  "swim",
        elevation = "elevation@PERMANENT",
        stations = "stations@PERMANENT",
        landuse = "landuse@PERMANENT",
        soil = "soil@PERMANENT",
        upthresh=40,
        lothresh=11,
    )
    files_created = ['file.cio', 'blank.str', 'file.cio',
                     'Sub/groundwater.tab', 'Sub/routing.tab',
                     'Sub/subbasin.tab']

    def test_session(self):
        from swimpy.grass import ProjectGrassSession
        self.project.settings(**self.grass_settings)
        with ProjectGrassSession(self.project, mapset='PERMANENT') as grass:
            rasts = grass.list_strings('rast')
            vects = grass.list_strings('vect')
            for m in ['elevation', 'landuse', 'soil']:
                self.assertIn(self.grass_settings[m], rasts)
            self.assertIn(self.grass_settings['stations'], vects)
        return

    def test_mswim_setup(self):
        files_created = [osp.join(self.project.projectdir, 'input', p)
                         for p in self.files_created]
        [os.remove(p) for p in files_created if osp.exists(p)]
        # needed settings
        self.project.settings(**self.grass_settings)
        # update subbasins (runs all other modules in postprocess)
        self.project.subbasins(verbose=False)
        for p in files_created:
            self.assertTrue(osp.exists(p))


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
