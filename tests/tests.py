#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `swimpy` package that are specific to the Blankenstein test case.

Project unspecific test are also included through inherited classes from the
swimpy.tests package that serves as a test suite to check the validity of any
project setup.
"""
from __future__ import print_function, absolute_import
import os
import os.path as osp
import sys
import subprocess
import shutil
import unittest
import cProfile, pstats

import pandas as pd
import pylab as pl

import swimpy
from swimpy.tests import test_project

SWIM_TEST_PROJECT = 'project/'
SWIM_REPO_PROJECT = '../dependencies/swim/project'

TEST_GRASSDB = 'grassdb'
MSWIM_GRASSDB = '../dependencies/m.swim/test/grassdb'

if not os.path.exists(SWIM_TEST_PROJECT):
    shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT)
if not os.path.exists(TEST_GRASSDB):
    shutil.copytree(MSWIM_GRASSDB, TEST_GRASSDB)


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

    def test_subcatch_parameters(self):
        from swimpy.input import subcatch_parameters as SubcatchParameters
        # read
        sbc = self.project.subcatch_parameters
        self.assertIsInstance(sbc, SubcatchParameters.plugin)
        BLKS = self.project.subcatch_parameters.loc['BLANKENSTEIN']
        self.assertIsInstance(BLKS, pd.Series)
        roc2 = self.project.subcatch_parameters['roc2']
        self.assertIsInstance(roc2, pd.Series)
        # write
        self.project.subcatch_parameters(roc2=1)
        self.assertEqual(self.project.subcatch_parameters['roc2'].mean(), 1)
        self.project.subcatch_parameters(BLANKENSTEIN=2)
        BLKS = self.project.subcatch_parameters.loc['BLANKENSTEIN'].mean()
        self.assertEqual(BLKS, 2)
        HOF = self.project.subcatch_parameters.loc['HOF']
        newparamdict = {'roc2': 3.0, 'roc4': 10.0}
        self.project.subcatch_parameters(HOF=newparamdict)
        for k, v in newparamdict.items():
            HOF[k] = v
        self.assertTrue((self.project.subcatch_parameters.loc['HOF'] ==
                         HOF).all())
        # write entire DataFrame
        self.project.subcatch_parameters(sbc.copy())
        nsbc = self.project.subcatch_parameters
        self.assertTrue((nsbc == sbc).all().all())

    def test_changed_parameters(self):
        verbose = False
        from random import random
        original = self.project.changed_parameters(verbose=verbose)
        bsn = self.project.basin_parameters
        scp = self.project.subcatch_parameters.T.stack().to_dict()
        nametags = [(k, None) for k in bsn] + list(scp.keys())
        nametags_original = [(e['name'], e['tags']) for e in original]
        for nt in nametags:
            self.assertIn(nt, nametags_original)
        run = self.project.browser.insert('run')
        for attr in original:
            self.project.browser.insert('parameter', run=run, **attr)
        self.project.basin_parameters(roc4=random(), da=random()*1000)
        changed = self.project.changed_parameters(verbose=verbose)
        self.assertEqual(sorted([e['name'] for e in changed]), ['da', 'roc4'])
        self.project.basin_parameters(**bsn)
        self.assertEqual(self.project.changed_parameters(verbose=verbose), [])
        self.project.subcatch_parameters(roc4=random())
        changed = self.project.changed_parameters(verbose=verbose)
        expresult = [('roc4', 'BLANKENSTEIN'), ('roc4', 'HOF')]
        nametags = sorted([(e['name'], e['tags']) for e in changed])
        self.assertEqual(nametags, expresult)


class TestProcessing(ProjectTestCase, test_project.Processing):

    def test_save_run(self):
        # test indicators and files
        indicators = ['indicator1', 'indicator2']
        ri_functions = [lambda p: 5,
                        lambda p: {'HOF': 0.1, 'BLANKENSTEIN': 0.2}]
        ri_values = {i: f(None) for i, f in zip(indicators, ri_functions)}
        files = ['file1', 'file2']
        somefile = osp.join(osp.dirname(__file__), 'tests.py')
        rf_functions = [lambda p: pd.DataFrame(list(range(100))),
                        lambda p: {'HOF': open(__file__),
                                   'BLANKENSTEIN': somefile}]
        rf_values = {i: f(None) for i, f in zip(files, rf_functions)}

        def check_files(fileobjects):
            self.assertEqual(len(fileobjects), 3)
            fdir = osp.join(self.project.browser.settings.filesdir, 'runs')
            for fo in fileobjects:
                self.assertTrue(osp.exists(fo.file.path))
                self.assertTrue(fo.file.path.startswith(fdir))
                self.assertIn(fo.tags.split()[0], files)
            return

        def check_indicators(indicatorobjects):
            self.assertEqual(len(indicatorobjects), 3)
            for io in indicatorobjects:
                self.assertIn(io.name, indicators)
            return
        # save run without any files or indicators
        run = self.project.save_run(notes='Some run notes',
                                    tags='testing test')
        self.assertIsInstance(run, self.project.browser.models['run'])
        self.assertTrue(hasattr(run, 'notes'))
        self.assertIn('test', run.tags.split())
        # pass indicators + files to save_run
        run = self.project.save_run(indicators=ri_values, files=rf_values)
        check_indicators(run.resultindicators.all())
        check_files(run.resultfiles.all())
        # pass as settings variables
        self.project.settings(**dict(zip(indicators, ri_functions)))
        self.project.settings(**dict(zip(files, rf_functions)))
        self.project.settings(resultfile_functions=files,
                              resultindicator_functions=indicators)
        run = self.project.save_run()
        check_indicators(run.resultindicators.all())
        check_files(run.resultfiles.all())


class TestRun(ProjectTestCase, test_project.Run):
    pass


def skip_if_py3(f):
    """Unittest skip test if PY3 decorator."""
    PY2 = sys.version_info < (3, 0)
    return f if PY2 else lambda self: print('not run in PY3.')


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

    class TestGrassTbl(swimpy.grass.GrassAttributeTable):
        vector = 'stations@PERMANENT'
        key = 'NAME'
        add_attributes = {'obs': {'HOF': pd.Series([12, 2, 2, 4])}}

    def test_session(self):
        from swimpy.grass import ProjectGrassSession
        self.project.settings(**self.grass_settings)
        with ProjectGrassSession(self.project, mapset='PERMANENT') as grass:
            rasts = grass.list_strings('rast')
            vects = grass.list_strings('vect')
            for m in ['elevation', 'landuse', 'soil']:
                self.assertIn(self.grass_settings[m].encode(), rasts)
            self.assertIn(self.grass_settings['stations'].encode(), vects)
        return

    @skip_if_py3
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

    @skip_if_py3
    def test_attribute_table(self):
        self.project.settings(self.TestGrassTbl, **self.grass_settings)
        self.assertTrue(hasattr(self.project, 'testgrasstbl'))
        self.assertIsInstance(self.project.testgrasstbl.obs.HOF, pd.Series)
        self.project.testgrasstbl['new'] = 1000
        self.project.testgrasstbl.write()
        self.project.testgrasstbl.read()
        self.assertEqual(self.project.testgrasstbl['new'].mean(), 1000)


class TestPlotting(ProjectTestCase):
    plot_prefix = 'plot'

    @property
    def plot_functions(self):
        fitems = self.project.settings.functions.items()
        fd = {n: f for n, f in fitems
              if n.split('.')[-1].startswith(self.plot_prefix)}
        return fd

    def test_output(self):
        print('Testing plot functions...')
        fig = pl.figure()
        for a, f in self.plot_functions.items():
            fig.clear()
            print(a)
            ppath = osp.join(self.project.projectdir, a+'.png')
            self.assertIsNotNone(self.project.settings[a](output=ppath))
            self.assertTrue(osp.exists(ppath))
        return


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
