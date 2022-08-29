"""A module for hydrology-related functionality.

Most functions are used in the `swimpy.output` module but they are placed here
to enable """
import warnings

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

    Arguments
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


def flow_duration(series, nbins=100):
    '''Cummulative flow duration from a pandas (discharge) series.

    Arguments
    ---------
    series : 1D-array
        Discharge series, nans will be removed.
    nbins : int
        How many regular bins to return.

    Returns
    -------
    pd.Series : Discharge value exceeded X percent (index) of time.
    '''
    ts = series[np.isfinite(series)]
    counts, bins = np.histogram(ts, bins=nbins, density=True)
    dense = counts * np.diff(bins)
    cumden = np.cumsum(dense[::-1])
    bins = bins[-2::-1]  # reverse and shorten by one
    series = pd.Series(bins, index=cumden*100)
    return series


def peak_over_threshold(data, percentile=1, threshold=None, maxgap=None):
    '''An efficient method to identify all peaks over a threshold (POT).

    Arguments
    ---------
    data : pd.Series
        The data series, preferable with a datetime/period index. If index
        is not datetime/period, daily frequency is assumed to calculate the
        recurrence interval (years).
    percentile : number
        The percentile threshold of data, e.g. 1 means Q1 (probability of exceedance).
    threshold : number, optional
        Absolute threshold to use for peak identification.
    maxgap : int, optional
        Largest gap between two threshold exceedance periods to count as single
        peak event. Number of timesteps. If not given, every exceedance is
        counted as individual peak event.

    Returns
    -------
    pd.DataFrame :
        Peak values ordered dataframe with order index and peak, length,
        peak date and recurrence columns.
    '''
    def find_steps(bools):
        # at each day before peak event start 1, at each last day a -1
        flgi = np.append(0, (bools[:-1].astype(int) - bools[1:].astype(int)))
        # remove -1
        flgi[flgi < 0] = 0
        # make steps
        return flgi.cumsum()
    # get peak groups index
    thresh = threshold or data.dropna().quantile(1 - percentile / 100.)
    ipeak = (data > thresh).values
    flgisteps = find_steps(ipeak)
    # pretend gaps smaller than maxgap are also peaks
    if maxgap:
        assert type(maxgap) == int
        gaps = data[~ipeak].groupby(flgisteps[~ipeak]).count()
        gaps[0] = maxgap
        ipeak[gaps[flgisteps] < maxgap] = True
        flgisteps = find_steps(ipeak)
    # get peaks
    qfl = data[ipeak]
    qpotgrp = qfl.groupby(flgisteps[ipeak])
    qpot = qpotgrp.max().to_frame('peak')
    qpot['length'] = qpotgrp.count()
    qpot['start_date'] = qpotgrp.apply(lambda df: df.index[0])
    qpot['peak_date'] = qpotgrp.idxmax()
    qpot['end_date'] = qpotgrp.apply(lambda df: df.index[-1])
    qpot.sort_values('peak', ascending=False, inplace=True)
    qpot.index = np.arange(1, len(qpot) + 1)
    nyears = (data.index.year[-1] - data.index.year[0] + 1
              if hasattr(data.index, 'year') else len(data.index) / 365.25)
    qpot['recurrence'] = float(nyears) / qpot.index
    return qpot


def gumbel_recurrence(q, recurrence):
    """Deprecated! Use dist_recurrence(..., dist='gumbel_r')."""
    warnings.warn(gumbel_recurrence.__doc__, DeprecationWarning)
    return dist_recurrence(q, recurrence, dist='gumbel_r')


def dist_recurrence(q, recurrence, dist='genextreme', shape=None, **fitkwargs):
    """Estimate values of given recurrence via any scipy distribution.

    Requires the `scipy` package to be installed.

    Arguments
    ---------
    q : 1D-array-like
        Observed values of distribution without NaNs. E.g. annual max Q.
    recurrence : 1D-array-like
        Recurrence intervals for which to return values for. Must be > 1.
    dist : str
        Any valid scipy distribution function. Common flood distributions are:
        genextreme, gumbel_r, weibull_min, lognorm, gamma. Refer to:
        https://docs.scipy.org/doc/scipy/reference/stats.html
    shape : float | None
        Fit distribution with fixed shape parameter or not if None. Only valid
        if dist actually has a shape parameter.
    fitkwargs :
        Keywords passed to the dist.fit method.
    """
    from scipy import stats
    assert hasattr(stats, dist), '%s is not a valid scipy distribution.' % dist
    df = getattr(stats, dist)
    kw = fitkwargs
    if shape and df.shapes:
        kw['fix_'+df.shapes] = shape
    # Estimate distribution parameters
    fits = df.fit(q, **kw)
    # quantiles of exceedence probability
    r = 1 - 1./recurrence
    # calculate recurrence/probability values
    ppf = df.ppf(r, *fits)
    return pd.Series(ppf, index=recurrence)


def hydrological_year_index(series, doy=None):
    """Return the series with an hydrological year index.

    Arguments
    ---------
    series : pd.Series
        Discharge time series with DateTime/Period index.
    doy : int, optional
        Day of year. If not given, the day with the lowest mean Q is used.
    """
    assert isinstance(series, pd.Series), \
        'series needs to be a pandas.Series instance.'
    assert hasattr(series.index, 'dayofyear'), \
        'Series index needs to be a DatetimeIndex or PeriodIndex.'
    if not doy:
        doy = series.groupby(series.index.dayofyear).mean().loc[:365].idxmin()
        print('Hydrological year starts on julian day %s' % doy)
    # build index
    years = series.index.year.values
    jdays = series.index.dayofyear.values - doy
    # shift year
    years[jdays < 0] = years[jdays < 0] - 1
    jdays[jdays < 0] = jdays[jdays < 0] + 365
    index = pd.MultiIndex.from_arrays([years, jdays], names=['year', 'day'])
    return pd.Series(series.values, index=index)
