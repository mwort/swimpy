"""A module for hydrology-related functionality.

Most functions are used in the `swimpy.output` module but they are placed here
to enable """
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


def peak_over_threshold(q, percentile=1, threshold=None, maxgap=None):
    '''An efficient method to identify all peaks over a threshold (POT).

    Arguments
    ---------
    q : pd.Series
        The discharge series, preferable with a datetime/period index. If index
        is not datetime/period, daily frequency is assumed to calculate the
        recurrence interval (years).
    percentile : number
        The percentile threshold of q., e.g. 1 means Q1.
    threshold : number, optional
        Absolute threshold to use for peak identification.
    maxgap : int, optional
        Largest gap between two threshold exceedance periods to count as single
        flood event. Number of timesteps. If not given, every exceedance is
        counted as individual flood event.

    Returns
    -------
    pd.DataFrame :
        Peak discharge ordered dataframe with order index and peak q, length,
        peak date and recurrence columns.
    '''
    def find_steps(bools):
        # at each day before flood start 1, at each last day a -1
        flgi = np.append(0, (bools[:-1].astype(int) - bools[1:].astype(int)))
        # remove -1
        flgi[flgi < 0] = 0
        # make steps
        return flgi.cumsum()
    # get flood groups index
    thresh = threshold or q.dropna().quantile(1 - percentile / 100.)
    iflood = (q > thresh).values
    flgisteps = find_steps(iflood)
    # pretend gaps smaller than maxgap are also floods
    if maxgap:
        assert type(maxgap) == int
        gaps = q[~iflood].groupby(flgisteps[~iflood]).count()
        gaps[0] = maxgap
        iflood[gaps[flgisteps] < maxgap] = True
        flgisteps = find_steps(iflood)
    # get floods
    qfl = q[iflood]
    qpotgrp = qfl.groupby(flgisteps[iflood])
    qpot = qpotgrp.max().to_frame('q')
    qpot['length'] = qpotgrp.count()
    qpot['start_date'] = qpotgrp.apply(lambda df: df.index[0])
    qpot['peak_date'] = qpotgrp.idxmax()
    qpot['end_date'] = qpotgrp.apply(lambda df: df.index[-1])
    qpot.sort_values('q', ascending=False, inplace=True)
    qpot.index = np.arange(1, len(qpot) + 1)
    nyears = (q.index.year[-1] - q.index.year[0] + 1
              if hasattr(q.index, 'year') else len(q.index) / 365.25)
    qpot['recurrence'] = float(nyears) / qpot.index
    return qpot


def gumbel_recurrence(q, recurrence):
    """Estimate values of given recurrence via the Gumbel distribution.

    Requires the `scipy` to be installed.

    Arguments
    ---------
    q : 1D-array-like
        Observed values of distribution. E.g. annual max Q.
    recurrence : 1D-array-like
        Recurrence intervals for which to return values for. Must be > 1.
    """
    from scipy.stats import genextreme as gev
    # Estimate Generalised Extreme Value distribution parameters
    fits = gev.fit(q, fix_c=0)
    # quantiles of exceedence probability
    r = 1 - 1./recurrence
    # calculate GEV values
    ppf = gev.ppf(r, *fits)
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
