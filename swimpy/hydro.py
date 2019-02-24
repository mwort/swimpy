"""A module for hydrology-related functions and classes."""

import numpy as np
import pandas as pd


def NSE(obs, sim):
    """Nash-Sutcliff-Efficiency.

    Arguments
    ---------
    obs, sim : same-length 1D array or pandas.Series
    """
    # subset to valid so that nans arent incl in sum, list to enforce obs and
    # sim are same length, throws error if not
    valid = np.isfinite(obs).tolist()
    # get sqared mean errors of obs-sim and obs-mean(obs)
    simSME = (obs[valid] - sim[valid])**2
    obsSME = (obs[valid] - obs[valid].mean())**2
    # calculate efficiency
    return 1 - (simSME.sum() / obsSME.sum())


def logNSE(obs, sim):
    '''Calculate log Nash-Sutcliffe-Efficiency through the NSE function

    Arguments
    ---------
    obs, sim : same-length 1D array or pandas.Series
    '''
    obs, sim = np.log(obs), np.log(sim)
    return NSE(obs, sim)


def mNSE(obs, sim):
    """Modified NSE weighted by Qobs to emphasise high/flood Q according to:
    Y. Hundecha, A. Bardossy (2004)


    Arguments
    ---------
    obs, sim : same-length 1D array or pandas.Series
    """
    valid = np.isfinite(obs).tolist()
    # get sqared mean errors of obs-sim and obs-mean(obs)
    simSME = obs[valid] * (obs[valid] - sim[valid])**2
    obsSME = obs[valid] * (obs[valid] - obs[valid].mean())**2
    # calculate efficiency
    return 1 - (simSME.sum() / obsSME.sum())


def pbias(obs, sim):
    """Calculate water balance of obs and sim in percent.

    Arguments
    ---------
    obs, sim : same-length 1D array or pandas.Series
    """
    valid = np.isfinite(obs).tolist()  # so that nans arent incl in sum
    return (sim[valid].sum() / obs[valid].sum() - 1) * 100


def q_to_runoff(q, area, freq='d'):
    '''Convert discharge volume [m^3s^-1] to height [mm].

    Arguments
    ---------
    q : scalar | array |  pd.Series | pd.DataFrame
        Discharge in m3s-2.
    area : scalar
        Drainage area in km2.
    freq : str
        Frequency of q, any of d, m, a. Derived from pandas period index if
        pandas object parsed.

    Returns
    -------
    <same type as q> : Runoff height in mm.
    '''
    periods = {'D': 24*60**2, 'M': 24*60**2*30, 'A': 24*60**2*365.24}
    try:
        time = q.index.freqstr[0]
    except AttributeError:
        time = freq.upper()
    qm = q*periods[time]   # in meter per unit time
    am = area*10**6        # in sq meter
    qmm = (qm/am)*10**3    # ratio in mm
    return qmm


def runoff_coefficient(q, p, area):
    '''Calculate runoff coefficient.

    Arguements
    ----------
    q : array |  pd.Series | pd.DataFrame
        Discharge Q[m3s-1]. If objects without PeriodIndex parsed, p is assumed
        to be annual sum and q annual mean discharge.
    p : <same as q>
        Precipitaton [mm]. Should have same frequency as q.
    area : scalar
        Drainage area in km2.

    Returns
    -------
    <same type as q> : Runoff height in mm.
    '''
    # convert Q m3 per second to mm over area
    try:
        f = q.index.freq
    except AttributeError:
        f = 'a'
    qtime = q_to_runoff(q, area, freq=f)
    rc = qtime / p
    return rc
