"""Tests for swimpy I/O: parameters, input files, output files.

Inherits project-independent mixin tests from swimpy/tests/.
"""
import os
import os.path as osp

import pandas as pd

from conftest import ProjectTestCase
from swimpy.tests import test_io, test_swimpy_config


class TestParameters(ProjectTestCase, test_io.Parameters):

    def test_catchment(self):
        scdef = self.project.catchment
        assert list(scdef.index) == ['BLANKENSTEIN', 'HOF']
        scdef.update(catchments=[1])
        assert list(scdef.index) == ['HOF']
        pars = scdef.loc['HOF'].drop('catchment_id').to_dict()
        assert pars == self.project.catchment_defaults
        scdef.update(subbasins=[1, 2])
        assert list(scdef.index) == ['BLANKENSTEIN']
        scdef.update()  # reset to original

    def test_changed_parameters(self):
        from random import random
        from numbers import Number
        self.project.browser.parameters.all().delete()
        original = self.project.changed_parameters(verbose=False)
        bsn = self.project.config_parameters.parlist
        scp = self.project.catchment.T.stack().to_dict()
        nametags = ([(k, None) for k, v in bsn.items() if isinstance(v, Number)]
                    + list(scp.keys()))
        nametags_original = [(e['name'], e['tags']) for e in original]
        for nt in nametags:
            assert nt in nametags_original
        run = self.project.browser.insert('run')
        for attr in original:
            self.project.browser.insert('parameter', run=run, **attr)
        self.project.config_parameters(iyr=1995, brunoffdat=True)
        changed = self.project.changed_parameters(verbose=False)
        assert sorted([e['name'] for e in changed]) == ['brunoffdat', 'iyr']
        self.project.config_parameters(**bsn)
        assert self.project.changed_parameters(verbose=False) == []
        self.project.catchment(roc4=random())
        changed = self.project.changed_parameters(verbose=False)
        expresult = [('roc4', 'BLANKENSTEIN'), ('roc4', 'HOF')]
        assert sorted([(e['name'], e['tags']) for e in changed]) == expresult


class TestInput(ProjectTestCase, test_io.Input, test_swimpy_config.Stations):

    def test_catchment(self):
        from swimpy.input import catchment as SubcatchParameters
        sbc = self.project.catchment
        assert isinstance(sbc, SubcatchParameters)
        assert isinstance(self.project.catchment.loc['BLANKENSTEIN'], pd.Series)
        assert isinstance(self.project.catchment['roc2'], pd.Series)
        # round-trip via CSV
        sbc.write('test.csv')
        sbc_test = self.project.catchment.read('test.csv')
        pd.testing.assert_frame_equal(sbc, sbc_test)
        os.remove('test.csv')
        # write via call
        self.project.catchment(roc2=1)
        assert self.project.catchment['roc2'].mean() == 1
        self.project.catchment['roc2'] = 2
        assert self.project.catchment['roc2'].mean() == 2
        self.project.catchment(BLANKENSTEIN=2)
        assert self.project.catchment.loc['BLANKENSTEIN'].mean() == 2
        HOF = self.project.catchment.loc['HOF']
        newparamdict = {'roc2': 3.0, 'roc4': 10.0}
        self.project.catchment(HOF=newparamdict)
        for k, v in newparamdict.items():
            HOF[k] = v
        assert (self.project.catchment.loc['HOF'] == HOF).all()
        # write entire DataFrame
        self.project.catchment(sbc.copy())
        assert (self.project.catchment.copy() == sbc.copy()).all().all()

    def test_discharge_write(self):
        self.project.discharge(stations=['HOF'])
        ro = self.project.discharge
        assert len(ro.columns) == 2
        assert 'HOF' in ro.columns
        ro(stations=['BLANKENSTEIN'])

    def test_netcdf_inputdata(self):
        import datetime as dt
        kw = dict(time=("1993", "1994-12-31"), subbasins=[1, 2, 3])
        p = self.project.climate.netcdf_inputdata.read("precipitation", **kw)
        assert p.shape == (365 * 2, 3)
        p = self.project.climate.netcdf_inputdata['tmean']
        nd = dt.date(2000, 12, 31) - dt.date(1990, 12, 31)
        assert p.shape == (nd.days, 11)
        clim = self.project.climate.netcdf_inputdata[["tmean", "tmin", "tmax"]]
        assert clim.shape == (nd.days, 11 * 3)
        assert len(clim.columns.levels) == 2

    def test_climcsv_inputdata(self):
        df = self.project.climate.inputdata
        df.write('test.csv')
        df_test = self.project.climate.inputdata.read('test.csv')
        pd.testing.assert_frame_equal(df, df_test)
        os.remove('test.csv')


class TestOutput(ProjectTestCase, test_io.Output):

    def test_output_attributes(self):
        from swimpy.output import OutputFile as ofileclass
        for k in self.project.output_files.keys():
            assert hasattr(self.project, k)
            assert isinstance(getattr(self.project, k), ofileclass)
        assert isinstance(
            self.project.output_files('cropland_daily_crop_out',
                                      'wb_daily_htp_prn'),
            list)
        assert len(
            self.project.subbasin_daily_discharge
            .loc['2000-12-27']['river_runoff']) == 11
        import pytest
        assert self.project.station_daily_discharge\
            .loc['1991-01-01']['discharge']['HOF'] == pytest.approx(5.047, abs=0.01)
        assert self.project.hydrotope_annual_gis\
            .loc[['1991']]['surface_runoff'].size == 182
        df = self.project.catchment_daily_bad_prn
        df.write('test.csv')
        df_test = self.project.catchment_daily_bad_prn.from_csv('test.csv')
        pd.testing.assert_frame_equal(df, df_test)
        os.remove('test.csv')

    def test_output_methods(self):
        for k in self.project.output_files.keys():
            assert hasattr(getattr(self.project, k), 'peak_over_threshold')
        dat = self.project.subbasin_daily_discharge\
            .peak_over_threshold(stations=1)
        assert dat.index.names == ['subbasin', 'variable', 'order']
        dat = self.project.station_daily_discharge\
            .peak_over_threshold(stations='BLANKENSTEIN', variables='discharge')
        assert dat.index.names == ['order']
        for m in ['obs_sim_overlap', 'NSE', 'rNSE', 'pbias', 'pbias_abs',
                  'plot_discharge_comparison']:
            assert hasattr(self.project.station_daily_discharge, m)
            assert not hasattr(self.project.hydrotope_annual_gis, m)
