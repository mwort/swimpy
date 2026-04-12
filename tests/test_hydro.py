"""Tests for the swimpy.hydro module (standalone, no project fixture needed)."""
import numpy as np
import pandas as pd
import pytest

from swimpy import hydro


@pytest.fixture
def obs_sim():
    """obs series with mean 1.5 and a nan hole, plus flat sim series."""
    obs = pd.Series(1, index=range(100), dtype=float)
    obs[50:] = 2
    obs[40:60] = np.nan
    sim = pd.Series(1.5, index=range(100))
    return obs, sim


def test_NSE(obs_sim):
    obs, sim = obs_sim
    assert hydro.NSE(obs, sim) == 0
    sim = sim.copy()
    sim[50:] = 2
    assert hydro.NSE(obs, sim) == pytest.approx(0.5)


def test_mNSE(obs_sim):
    obs, sim = obs_sim
    assert hydro.mNSE(obs, sim) == 0
    sim = sim.copy()
    sim[50:] = 2
    assert hydro.mNSE(obs, sim) == pytest.approx(2.0 / 3)


def test_pbias(obs_sim):
    obs, sim = obs_sim
    assert hydro.pbias(obs, sim) == 0
    sim = sim * 1.1
    assert hydro.pbias(obs, sim) == pytest.approx(10)


def test_dist_recurrence():
    sim = np.random.rand(1000)
    rec = hydro.dist_recurrence(sim, 100.0 / np.arange(1, 11))
    assert rec[10.0] < rec[100.0]
