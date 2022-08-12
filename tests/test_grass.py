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

    files_created = ['subbasin.csv', 'subbasin_routing.csv', 'hydrotope.csv', 'catchment.csv']

    class grassattrtbl(mmgrass.GrassAttributeTable):
        vector = 'stations@PERMANENT'
        key = 'NAME'
        obs = pd.DataFrame({'HOF': [12, 2, 2, 4]})

    def test_session(self):
        with mmgrass.GrassSession(self.project, mapset='PERMANENT') as grass:
            rasts = grass.list_strings('rast')
            vects = grass.list_strings('vect')
        self.assertIn(self.project.grass_setup['landuse_id'], rasts)
        self.assertIn(self.project.grass_setup['soil_id'], rasts)
        self.assertIn(self.project.grass_setup['elevation'], rasts)
        self.assertIn(self.project.grass_setup['stations'], vects)
        return

    def test_mswim_setup(self):
        files_created = [osp.join(self.project.projectdir, 'input', p)
                         for p in self.files_created]
        [os.remove(p) for p in files_created if osp.exists(p)]
        # update subbasins (runs all other modules in postprocess)
        self.project.subbasin.update(verbose=False)
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

    def test_to_grass(self):
        hyd_file = 'hydrotope_annual_gis'
        sub_file = 'subbasin_daily_river_discharge'

        with mmgrass.GrassOverwrite(verbose=False):
            getattr(self.project, hyd_file).to_grass(
                variable=['surface_runoff', 'crop_yield'],
                timestep=slice('1991', '1995'),
                mapset=hyd_file)
            getattr(self.project, sub_file).to_grass(
                variable='discharge', mapset=sub_file,
                timestep=slice('1991-01-01', '1991-01-10'))
        for f in [hyd_file, sub_file]:
            with mmgrass.GrassSession(self.project, mapset=f) as grass:
                rasters = grass.list_strings('raster', f+'*', mapset=f)
                self.assertEqual(len(rasters), 10)


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
