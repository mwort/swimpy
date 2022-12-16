import numpy as np


def cached(method):
    def wrapped(self):
        mname = '_'+method.__name__
        if not hasattr(self, mname):
            setattr(self, mname, method(self))
        return getattr(self, mname)
    return wrapped


def pbias(a, b):
    """Absolute percent bias between value a and b."""
    return np.abs((a/b - 1) * 100)


class WaterBalance:

    @cached
    def subbasin_area(self):
        sba = self.project.hydrotope.groupby('subbasin_id')['area']
        return sba.sum()

    @cached
    def catchment_daily_precipitation_input(self):
        totpsb = self.project.climate.inputdata.precipitation
        sba = self.subbasin_area()
        totpc = totpsb.mul(sba/sba.sum(), axis=1).sum(axis=1)
        return totpc

    # TODO: catchment_annual_* file with precipitation, river_runoff, eta
    @cached
    def catchment_annual_waterbalance(self):
        df = self.project.catchment_daily_bad_prn
        df_catch = df.iloc[:,df.columns.get_level_values(1) == 0]
        return df_catch.resample('a').sum()

    def test_catchment_annual_precipitation_output(self):
        """Fails if precipitation correction is active (xgrad1 != 0)"""
        p_in = self.catchment_daily_precipitation_input().resample('a').sum()
        p_out = self.catchment_annual_waterbalance()['precipitation']
        msg = 'Input precipitation unequal output precipitation in year %s'
        for y in p_out.index:
            self.assertAlmostEqual(p_in[y], p_out[y], 2, msg=msg % y)

    def test_catchment_annual_3qs_eta(self):
        """Missing components: snow weq., ..."""
        accept_dev = 1
        wb = self.catchment_annual_waterbalance().sum()
        runeta = wb['river_runoff'] + wb['eta']
        msg = ('River runoff + eta is deviating by more than %d%% from total precipitation.'
               % accept_dev)
        self.assertLess(pbias(runeta, wb['precipitation']), accept_dev, msg=msg)

    def test_catchment_total_discharge(self):
        """Missing components: transmission losses, snow weq., deep groundwater
        losses, soil water."""
        accept_dev = 3
        a = self.subbasin_area().sum()
        p_vol = self.catchment_daily_precipitation_input().sum()*1e-3*a
        eta_vol = self.catchment_annual_waterbalance()['eta'].sum()*1e-3*a
        q_volume = self.project.subbasin_daily_river_discharge['discharge'][1].sum()*24*60**2
        msg = ('Total discharge is deviating by more than %d%% from total '
               'total precipitation - eta.' % accept_dev)
        self.assertLess(pbias(p_vol-eta_vol, q_volume), accept_dev, msg=msg)

    def test_catchment_discharge_eq_3q(self):
        """Missing components: transmission losses, initial reservoir water,
        ...?"""
        accept_dev = 5
        wb = self.catchment_annual_waterbalance().sum()
        vol3q = wb['river_runoff']*1e-3*self.subbasin_area().sum()
        volq = self.project.subbasin_daily_discharge['discharge'][1].sum()*24*60**2
        msg = ("Total discharge is deviating by more than %d%% from river runoff"
               % accept_dev)
        self.assertLess(pbias(vol3q, volq), accept_dev, msg=msg)
