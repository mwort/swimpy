#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the grass linking of `swimpy` using the Blankenstein test case.
"""
from __future__ import print_function, absolute_import
import os
import os.path as osp
import sys
import unittest
import cProfile, pstats

import pandas as pd
from modelmanager.plugins import grass as mmgrass

from tests import ProjectTestCase


def skip_if_py3(f):
    """Unittest skip test if PY3 decorator."""
    PY2 = sys.version_info < (3, 0)
    return f if PY2 else lambda self: print('not run in PY3.')


class TestGrass(ProjectTestCase):

    files_created = ['file.cio', 'blank.str', 'file.cio',
                     'Sub/groundwater.tab', 'Sub/routing.tab',
                     'Sub/subbasin.tab']

    class grassattrtbl(mmgrass.GrassAttributeTable):
        vector = 'stations@PERMANENT'
        key = 'NAME'
        obs = pd.DataFrame({'HOF': [12, 2, 2, 4]})

    def test_session(self):
        with mmgrass.GrassSession(self.project, mapset='PERMANENT') as grass:
            rasts = grass.list_strings('rast')
            vects = grass.list_strings('vect')
        self.assertIn(self.project.grass_setup['landuse'], rasts)
        self.assertIn(self.project.grass_setup['soil'], rasts)
        self.assertIn(self.project.grass_setup['elevation'], rasts)
        self.assertIn(self.project.grass_setup['stations'], vects)
        return

    def test_mswim_setup(self):
        files_created = [osp.join(self.project.projectdir, 'input', p)
                         for p in self.files_created]
        [os.remove(p) for p in files_created if osp.exists(p)]
        # update subbasins (runs all other modules in postprocess)
        self.project.subbasins(verbose=False)
        for p in files_created:
            self.assertTrue(osp.exists(p))

    def test_attribute_table(self):
        self.project.settings(self.grassattrtbl)
        self.assertTrue(hasattr(self.project, 'grassattrtbl'))
        self.assertIsInstance(self.project.grassattrtbl.obs.HOF, pd.Series)
        self.project.grassattrtbl['new'] = 1000
        self.project.grassattrtbl.write()
        self.project.grassattrtbl.read()
        self.assertEqual(self.project.grassattrtbl['new'].mean(), 1000)

    def test_to_raster(self):
        hyd_file = 'hydrotope_annual_evapotranspiration_actual'
        sub_file = 'subbasin_daily_waterbalance'
        with mmgrass.GrassOverwrite(verbose=False):
            getattr(self.project, hyd_file).to_raster(mapset=hyd_file)
            ts = slice('1991-01-01', '1991-01-10')
            getattr(self.project, sub_file).to_raster(
                'AET', mapset=sub_file, timestep=ts)
        for f in [hyd_file, sub_file]:
            with mmgrass.GrassSession(self.project, mapset=f) as grass:
                rasters = grass.list_strings('raster', f+'*', mapset=f)
                self.assertEqual(len(rasters), 10)


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
