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
# project-independent tests
from swimpy.tests import (test_io, test_running, test_swimpy_config,
                          test_waterbalance)

SWIM_REPO = '../dependencies/swim'

SWIM_TEST_PROJECT = 'project/'
SWIM_REPO_PROJECT = SWIM_REPO+'/project'

TEST_GRASSDB = 'grassdb'
MSWIM_GRASSDB = '../dependencies/m.swim/test/grassdb'

TEST_SETTINGS = './test_settings.py'


class TestSetup(unittest.TestCase):

    projectdir = SWIM_TEST_PROJECT
    resourcedir = osp.join(projectdir, 'swimpy')

    def test_setup(self):
        self.project = swimpy.project.setup(self.projectdir, name='test',
                                            gitrepo=SWIM_REPO)
        self.assertTrue(isinstance(self.project, swimpy.Project))
        self.assertTrue(osp.exists(self.resourcedir))
        self.assertTrue(osp.exists(self.project.resourcedir))
        self.assertTrue(osp.exists(osp.join(self.projectdir, 'input')))
        self.assertTrue(osp.exists(osp.join(self.projectdir, 'test.nml')))
        self.assertTrue(osp.islink(osp.join(self.projectdir, 'swim')))
        for a in ['browser', 'clone', 'templates']:
            self.assertTrue(hasattr(self.project, a))
            self.assertIsNot(getattr(self.project, a), None)

    def test_setup_commandline(self):
        subprocess.call(['swimpy', 'setup', '--name=test',
                         '--projectdir='+self.projectdir,
                         '--gitrepo='+SWIM_REPO])
        self.assertTrue(osp.exists(self.resourcedir))
        self.project = swimpy.Project(self.projectdir)

    def tearDown(self):
        self.project.browser.settings.unset()
        shutil.rmtree(self.projectdir)


class ProjectTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # copy swim and m.swim test projects
        shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT)
        shutil.copytree(MSWIM_GRASSDB, TEST_GRASSDB)
        # new test project with Blankenstein project
        p = swimpy.project.setup(SWIM_TEST_PROJECT)
        # add test_settings.py
        shutil.copy(TEST_SETTINGS, p.settings.file)
        self.project = swimpy.Project(SWIM_TEST_PROJECT)
        # reload brower project instance
        self.project.browser.project.settings.load()

    @classmethod
    def tearDownClass(self):
        self.project.browser.settings.unset()
        shutil.rmtree(self.project.projectdir)
        shutil.rmtree(TEST_GRASSDB)


class TestParameters(ProjectTestCase, test_io.Parameters):

    def test_catchment(self):
        scdef = self.project.catchment
        self.assertEqual(list(scdef.index), ['BLANKENSTEIN', 'HOF'])
        scdef.update(catchments=[1])
        self.assertEqual(list(scdef.index), ['HOF'])
        pars = scdef.loc['HOF']
        pars = pars.drop('catchment_id')
        pars = pars.to_dict()
        self.assertEqual(pars, self.project.catchment_defaults)
        scdef.update(subbasins=[1, 2])
        self.assertEqual(list(scdef.index), ['BLANKENSTEIN'])
        scdef.update()  # reset to original

    def test_changed_parameters(self):
        verbose = False
        from random import random
        original = self.project.changed_parameters(verbose=verbose)
        bsn = self.project.config_parameters.parlist
        scp = self.project.catchment.T.stack().to_dict()
        nametags = [(k, None) for k in bsn] + list(scp.keys())
        nametags_original = [(e['name'], e['tags']) for e in original]
        for nt in nametags:
            self.assertIn(nt, nametags_original)
        run = self.project.browser.insert('run')
        for attr in original:
            self.project.browser.insert('parameter', run=run, **attr)
        self.project.config_parameters(iyr=1995, bsubcatch=False)
        changed = self.project.changed_parameters(verbose=verbose)
        self.assertEqual(sorted([e['name'] for e in changed]), ['bsubcatch', 'iyr'])
        self.project.config_parameters(**bsn)
        self.assertEqual(self.project.changed_parameters(verbose=verbose), [])
        self.project.catchment(roc4=random())
        changed = self.project.changed_parameters(verbose=verbose)
        expresult = [('roc4', 'BLANKENSTEIN'), ('roc4', 'HOF')]
        nametags = sorted([(e['name'], e['tags']) for e in changed])
        self.assertEqual(nametags, expresult)


class TestInput(ProjectTestCase, test_io.Input, test_swimpy_config.Stations):

    def test_catchment(self):
        from swimpy.input import catchment as SubcatchParameters
        # read
        sbc = self.project.catchment
        self.assertIsInstance(sbc, SubcatchParameters)
        BLKS = self.project.catchment.loc['BLANKENSTEIN']
        self.assertIsInstance(BLKS, pd.Series)
        roc2 = self.project.catchment['roc2']
        self.assertIsInstance(roc2, pd.Series)
        # make sure write function works as expected
        sbc.write('test.csv')
        sbc_test = self.project.catchment.read('test.csv')
        pd.testing.assert_frame_equal(sbc, sbc_test)
        os.remove('test.csv')
        # write
        self.project.catchment(roc2=1)
        self.assertEqual(self.project.catchment['roc2'].mean(), 1)
        self.project.catchment['roc2'] = 2
        self.assertEqual(self.project.catchment['roc2'].mean(), 2)
        self.project.catchment(BLANKENSTEIN=2)
        BLKS = self.project.catchment.loc['BLANKENSTEIN'].mean()
        self.assertEqual(BLKS, 2)
        HOF = self.project.catchment.loc['HOF']
        newparamdict = {'roc2': 3.0, 'roc4': 10.0}
        self.project.catchment(HOF=newparamdict)
        for k, v in newparamdict.items():
            HOF[k] = v
        self.assertTrue((self.project.catchment.loc['HOF'] ==
                         HOF).all())
        # write entire DataFrame
        self.project.catchment(sbc.copy())
        nsbc = self.project.catchment
        self.assertTrue((nsbc.copy() == sbc.copy()).all().all())

    def test_discharge_write(self):
        self.project.discharge(stations=['HOF'])
        ro = self.project.discharge
        self.assertEqual(len(ro.columns), 2)
        self.assertIn('HOF', ro.columns)
        ro(stations=['BLANKENSTEIN'])

    def test_netcdf_inputdata(self):
        import datetime as dt
        kw = dict(time=("1993", "1994-12-31"), subbasins=[1, 2, 3])
        p = self.project.climate.netcdf_inputdata.read("precipitation", **kw)
        self.assertEqual(p.shape, (365*2, 3))
        p = self.project.climate.netcdf_inputdata['tmean']
        nd = dt.date(2000, 12, 31)-dt.date(1990, 12, 31)
        self.assertEqual(p.shape, (nd.days, 11))
        clim = self.project.climate.netcdf_inputdata[["tmean", "tmin", "tmax"]]
        self.assertEqual(clim.shape, (nd.days, 11*3))
        self.assertEqual(len(clim.columns.levels), 2)
    
    def test_climcsv_inputdata(self):
        # make sure write function works as expected
        df = self.project.climate.inputdata
        df.write('test.csv')
        df_test = self.project.climate.inputdata.read('test.csv')
        pd.testing.assert_frame_equal(df, df_test)
        # clean up
        os.remove('test.csv')


class TestProcessing(ProjectTestCase, test_running.Cluster):

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
        check_indicators(run.indicators.all())
        check_files(run.files.all())
        # pass as settings variables
        self.project.settings(**dict(zip(indicators, ri_functions)))
        self.project.settings(**dict(zip(files, rf_functions)))
        self.project.settings(save_run_files=files,
                              save_run_indicators=indicators)
        run = self.project.save_run()
        check_indicators(run.indicators.all())
        check_files(run.files.all())
    
    def test_run(self):
        # check that swim has run via logfile
        logfile = osp.join(self.project.outputpath, 'swim.log')
        os.remove(logfile)
        self.project.run(save=False, quiet=True)
        self.assertTrue(osp.exists(logfile))


# TODO: needs revision; how to test plot functions of plugins ('OutputFile')?
# class TestOutputPlotting(ProjectTestCase):

#     plot_prefix = 'plot'
#     default_positional_arguments = {
#         'station': 'HOF'
#     }

#     @property
#     def plot_functions(self):
#         pset = self.project.settings
#         fd = []
#         for n in pset.functions.keys():
#             prts = n.split('.')
#             if prts[-1].startswith(plot_prefix):
#                 pim = pset.plugins.get('.'.join(prts[:-1]), type).__module__
#                 if pim == 'swimpy.output':
#                     fd += [n]
#         return fd

#     def run_with_defaults(self, fname, **kwargs):
#         panames = self.project.settings.functions[fname].positional_arguments
#         pargs = [self.default_positional_arguments[a] for a in panames]
#         return self.project.settings[fname](*pargs, **kwargs)

#     def test_output(self):
#         print('Testing plot functions...')
#         fig = pl.figure()
#         for a in self.plot_functions:
#             fig.clear()
#             print(a)
#             ppath = osp.join(self.project.projectdir, a+'.png')
#             self.assertIsNotNone(self.run_with_defaults(a, output=ppath))
#             self.assertTrue(osp.exists(ppath))
#         return

#     def test_runs(self):
#         resfile_interfaces = self.project.output_interfaces
#         setprops = self.project.settings.properties
#         resfile_plotf = []
#         for n in self.plot_functions:
#             ifn = '.'.join(n.split('.')[:-1])
#             if ifn in resfile_interfaces and setprops[ifn].plugin.path:
#                 resfile_plotf.append(n)
#         resfiles_w_plotf = ['.'.join(n.split('.')[:-1]) for n in resfile_plotf]
#         resfile_plotf.append('plot_summary')
#         run = self.project.save_run(files=resfiles_w_plotf,
#                                     notes='TestPlotting.test_runs')
#         fig = pl.figure()
#         for a in resfile_plotf:
#             print(a)
#             self.assertIsNotNone(self.run_with_defaults(a, runs=('*', run)))
#             fig.clear()
#         return


class TestOutput(ProjectTestCase, test_io.Output):
    
    def test_output_attributes(self):
        # all defined output files are project attributes
        from swimpy.output import OutputFile as ofileclass
        for k in self.project.output_files.keys():
            self.assertTrue(hasattr(self.project, k))
            self.assertIsInstance(getattr(self.project, k),
                                  ofileclass)
        # get / indexing methods (format of underlying pd.DataFrame)
        self.assertIsInstance(self.project.output_files('hydrotope_label_daily_crop_out',
                                                        'hydrotope_label_daily_htp_prn'),
                              list)
        self.assertEqual(len(self.project.subbasin_daily_river_discharge.loc['2000-12-27']['river_runoff']), 11)
        self.assertAlmostEqual(self.project.subbasin_label_daily_selected_stations_discharge.loc['1991-01-01']['discharge']['HOF'], 5.187)
        self.assertEqual(self.project.hydrotope_annual_gis.loc[['1991']]['surface_runoff'].size, 182)
        # make sure write function works as expected
        df = self.project.catchment_daily_bad_prn
        df.write('test.csv')
        df_test = self.project.catchment_daily_bad_prn.from_csv('test.csv')
        pd.testing.assert_frame_equal(df, df_test)
        # clean up
        os.remove('test.csv')


# class TestWaterBalance(ProjectTestCase, test_waterbalance.WaterBalance):
#     pass


if __name__ == '__main__':
    # just setup test files/directories
    if len(sys.argv) == 2 and sys.argv[1] == 'setup':
        ProjectTestCase.setUpClass()
        sys.exit(0)
    # run unittest
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
