#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the `swimpy.hydro` module.
"""
import unittest
import cProfile, pstats

import numpy as np
import pandas as pd

from swimpy import hydro


class TestHydro(unittest.TestCase):

    def obs_sim_data(self):
        """Create obs series with mean 1.5 and a nan hole and sim series."""
        obs = pd.Series(1, index=range(100))
        obs[50:] = 2
        obs[40:60] = np.nan
        sim = pd.Series(1.5, index=range(100))
        return obs, sim

    def test_NSE(self):
        obs, sim = self.obs_sim_data()
        self.assertEqual(hydro.NSE(obs, sim), 0)
        sim[50:] = 2
        self.assertAlmostEqual(hydro.NSE(obs, sim), 0.5)

    def test_mNSE(self):
        obs, sim = self.obs_sim_data()
        self.assertEqual(hydro.mNSE(obs, sim), 0)
        sim[50:] = 2
        self.assertAlmostEqual(hydro.mNSE(obs, sim), 2./3)

    def test_pbias(self):
        obs, sim = self.obs_sim_data()
        self.assertEqual(hydro.pbias(obs, sim), 0)
        sim = sim * 1.1
        self.assertAlmostEqual(hydro.pbias(obs, sim), 10)


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
