"""Tests for the GRASS GIS integration of swimpy."""
import os
import os.path as osp

import shutil

import pandas as pd
import pytest
from modelmanager.plugins import grass as mmgrass

if shutil.which('grass') is None:
    pytest.skip('GRASS GIS not found on PATH', allow_module_level=True)

from conftest import ProjectTestCase


@pytest.mark.grass
class TestGrass(ProjectTestCase):

    files_created = ['subbasin.csv', 'subbasin_routing.csv',
                     'hydrotope.csv', 'catchment.csv']

    class grassattrtbl(mmgrass.GrassAttributeTable):
        vector = 'stations@PERMANENT'
        key = 'NAME'
        obs = pd.DataFrame({'HOF': [12, 2, 2, 4]})

    def test_session(self):
        with mmgrass.GrassSession(self.project, mapset='PERMANENT') as grass:
            rasts = grass.list_strings('rast')
            vects = grass.list_strings('vect')
        assert self.project.grass_setup['landuse_id'] in rasts
        assert self.project.grass_setup['soil_id'] in rasts
        assert self.project.grass_setup['elevation'] in rasts
        assert self.project.grass_setup['stations'] in vects

    def test_mswim_setup(self):
        files_created = [osp.join(self.project.projectdir, 'input', p)
                         for p in self.files_created]
        for p in files_created:
            if osp.exists(p):
                os.remove(p)
        self.project.subbasin.update(verbose=False)
        for p in files_created:
            assert osp.exists(p)

    def test_attribute_table(self):
        self.project.settings(self.grassattrtbl)
        assert hasattr(self.project, 'grassattrtbl')
        assert isinstance(self.project.grassattrtbl.obs.HOF, pd.Series)
        self.project.grassattrtbl['new'] = 1000
        self.project.grassattrtbl.write()
        self.project.grassattrtbl.read()
        assert self.project.grassattrtbl['new'].mean() == 1000

    def test_to_grass(self):
        hyd_file = 'hydrotope_annual_gis'
        sub_file = 'subbasin_daily_discharge'
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
                rasters = grass.list_strings('raster', f + '*', mapset=f)
                assert len(rasters) == 10
