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
        sba = self.project.hydrotopes.attributes.groupby('subbasinID')['area']
        return sba.sum()

    @cached
    def catchment_daily_precipitation_input(self):
        totpsb = self.project.climate.inputdata.precipitation
        sba = self.subbasin_area()
        totpc = totpsb.mul(sba/sba.sum(), axis=1).sum(axis=1)
        return totpc

    @cached
    def catchment_annual_waterbalance(self):
        return self.project.catchment_annual_waterbalance

    def test_catchment_annual_precipitation_output(self):
        """Fails if precipitation correction is active (xgrad1 != 0)"""
        p_in = self.catchment_daily_precipitation_input().resample('a').sum()
        p_out = self.catchment_annual_waterbalance()['PREC']
        msg = 'Input precipitation unequal output precipitation in year %s'
        for y in p_out.index:
            self.assertAlmostEqual(p_in[y], p_out[y], 2, msg=msg % y)

    def test_catchment_annual_3qs_eta(self):
        """Missing components: snow weq., ..."""
        accept_dev = 1
        wb = self.catchment_annual_waterbalance().sum()
        msg = ('Total 3Q+AET is deviating by more than %d%% from total PREC.'
               % accept_dev)
        self.assertLess(pbias(wb['3Q+AET'], wb['PREC']), accept_dev, msg=msg)

    def test_catchment_total_discharge(self):
        """Missing components: transmission losses, snow weq., deep groundwater
        losses, soil water."""
        accept_dev = 3
        a = self.subbasin_area().sum()
        p_vol = self.catchment_daily_precipitation_input().sum()*1e-3*a
        eta_vol = self.catchment_annual_waterbalance()['AET'].sum()*1e-3*a
        q_volume = self.project.subbasin_daily_discharge[1].sum()*24*60**2
        msg = ('Total discharge is deviating by more than %d%% from total '
               'total PREC - AET.' % accept_dev)
        self.assertLess(pbias(p_vol-eta_vol, q_volume), accept_dev, msg=msg)

    def test_catchment_discharge_eq_3q(self):
        """Missing components: transmission losses, initial reservoir water,
        ...?"""
        accept_dev = 5
        wb = self.catchment_annual_waterbalance().sum()
        vol3q = wb['3Q']*1e-3*self.subbasin_area().sum()
        volq = self.project.subbasin_daily_discharge[1].sum()*24*60**2
        msg = ("Total discharge is deviating by more than %d%% from 3Q runoff"
               % accept_dev)
        self.assertLess(pbias(vol3q, volq), accept_dev, msg=msg)
