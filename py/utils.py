
from __future__ import division

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ts2dt(ts):
    '''Convert int timestamp to datetime'''
    # Source: https://stackoverflow.com/questions/3694487/initialize-a-datetime-object-with-seconds-since-epoch
    # return np.array(pd.to_datetime(ts, unit='s', utc=True).tz_convert('Europe/London'))
    if isinstance(ts, int):
        return datetime.fromtimestamp(ts)
    else:
        return [ts2dt(i) for i in ts]


def dt2ts(dt):
    '''Convert datetime to int timestamp'''
    # Source: https://stackoverflow.com/questions/11743019/convert-python-datetime-to-epoch-with-strftime
    if isinstance(dt, datetime):
        return int(dt.strftime('%s'))
    else:
        return [dt2ts(i) for i in dt]


def test_ts_dt_conversions(v1=1381323977):
    v2 = dt2ts(ts2dt(v1))
    if v1 == v2:
        return 'Values are the same'
    else:
        return 'Values are different: started as {} and ended as {}'.format(ts2dt(v1), ts2dt(v2))


def hours_in_day(dt):
    return int((dt2ts(dt + timedelta(days=1)) - dt2ts(dt)) / 3600)


def date_to_datetime(d):
    return datetime.combine(d, datetime.min.time())


def floor_time(dt):
    return datetime.combine(dt.date(), datetime.min.time())


def dt64_to_datetime(dt64):
    # dt64 is numpy datetime format.
    ns = 1e-9 # number of seconds in a nanosecond
    return datetime.utcfromtimestamp(dt64.astype(int) * ns)


def array_to_1d(x):
    return np.squeeze(np.asarray(x))


def calc_diff(df_col, pad_end=True, nan_fill=None):
    '''
    Calculate diff of array of ints, padding beginning or end with nan_fill
    '''
    beg = df_col.diff()[1:]
    end = nan_fill
    if not pad_end:
        beg, end = end, beg  # pad beginning instead
    return np.append(beg, end).astype(int)


def align_arrays(actual, desired, padder=None, is_debug=False):
    '''
    Returns indices idx of actual array such that actual[idx[i]] is that maximum
    value of actual such that actual[idx[i]] <= desired[idx[i]] for all i.
    '''
    a = 0
    idx = []
    
    # Convenience debug function.
    def debug():
        if is_debug:
            print 'a = {}, d = {}, actual[a] = {}, desired[d] = {}, idx = {}'.format(a, d, actual[a], desired[d], idx)

    for d in range(len(desired)):
        while a <= len(actual)-1 and actual[a] <= desired[d]:
            debug()
            a += 1
        idx.append(a-1)
    
    # Reassign idx values of -1 (in case where actual[0] > desired[d] for beginning d) to padder value.
    for i in range(len(desired)):
        if idx[i] == -1:
            idx[i] = padder
        else:
            break
    
    return idx


def plot_series(x):
    return pd.Series(x, index=range(len(x))).plot()


def plot_empir_cum(x):
    return plt.step(sorted(x), np.arange(len(x))/len(x), color='black')


class MultinomialSampler(object):
    """
    Fast (O(log n)) sampling from a discrete probability
    distribution, with O(n) set-up time.
    """

    def __init__(self, p, verbose=False):
        # n = len(p)
        p = p.astype(float) / sum(p)
        self._cdf = np.cumsum(p)

    def sample(self, k=1):
        rs = np.random.random(k)
        # binary search to get indices
        return np.searchsorted(self._cdf, rs)

    def __call__(self, **kwargs):
        return self.sample(**kwargs)

    def reconstruct_p(self):
        """
        Return the original probability vector.
        Helpful for debugging.
        """
        n = len(self._cdf)
        p = np.zeros(n)
        p[0] = self._cdf[0]
        p[1:] = (self._cdf[1:] - self._cdf[:-1])
        return p

def multinomial_sample(p):
    """
    Wrapper to generate a single sample,
    using the above class.
    """
    return MultinomialSampler(p).sample(1)[0]