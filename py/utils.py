
from __future__ import division

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


def blockPrint():
    # https://stackoverflow.com/a/8391735/4794432
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    # https://stackoverflow.com/a/8391735/4794432
    sys.stdout = sys.__stdout__


def makedirs2(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


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


def timedelta64_to_secs(timedelta):
    """
    Credit: Jack Kelly and NILMTK package:
    http://nilmtk.github.io/nilmtk/v0.1.1/_modules/nilmtk/utils.html#timedelta64_to_secs

    Convert `timedelta` to seconds.

    Parameters
    ----------
    timedelta : np.timedelta64

    Returns
    -------
    float : seconds
    """
    if len(timedelta) == 0:
        return np.array([])
    else:
        return timedelta / np.timedelta64(1, 's')


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


def moving_avg(x, window_width):
    '''
    Calculate moving average.
    Source: https://stackoverflow.com/a/34387987/4794432
    '''
    cumsum_vec = np.cumsum(np.insert(x, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec


def calc_loss(y, y_hat, loss_type):
    '''
    Calculates MSE and standard error.
    '''
    assert len(y)==len(y_hat)
    assert loss_type in ['mse', 'mae']
    y, y_hat = np.array(y), np.array(y_hat)
    N = len(y)
    if loss_type=='mse':
        L = (y - y_hat)**2
    elif loss_type=='mae':
        L = np.abs(y - y_hat)
    else:
        raise ValueError('loss_type not recognized')
    se = np.sqrt(np.var(L) / N)
    return L, np.mean(L), se


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


def change_array_size(a, desired_len):
    '''
    Expands or contracts array (by adding or removing elements) to have a length of
    desired_len.
    '''
    assert isinstance(a, np.ndarray)
    ratio = len(a) / desired_len
    idx = align_arrays(range(len(a)), np.arange(desired_len)*ratio)
    return a[idx]


def plot_series(x, figsize=(11,2)):
    return pd.Series(x, index=range(len(x))).plot(figsize=figsize)


def plot_empir_cum(x):
    '''Plot empirical cumulative distribution'''
    # https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
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


def weighted_choice(choices):
    '''
    Chooses randomly among alternatives given list of tuples (option, weight).
    https://stackoverflow.com/a/3679747/4794432
    '''
    total = sum(w for c, w in choices)
    r = np.random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def repeats_above_value(x, val, get_longest_only=False):
    '''
    Get number of elements above x that are greater than val and same as the previous
    data element. get_longest_only only calculates the longest stretch of repeats
    instead of all of them.
    '''

    # # Slower but simpler version. Calculates all values in run,
    # # not just the repeats.
    # counter = 0
    # max_count = 0
    # for i in range(1, len(x)):
    #     if x[i] == x[i-1] and x[i] > val:
    #         counter += 1
    #         if counter > max_count:
    #             max_count = counter
    #     else:
    #         counter = 0
    # return max_count

    mask_above_val = x > val
    mask_diff = np.concatenate([[False], np.diff(x) == 0])
    idx = np.where(mask_above_val & mask_diff)[0]
    
    if get_longest_only:
        
        # idx = np.where(mask_above_val & mask_diff)[0]
        if len(idx)==0:
            return 0

        counter = 1
        max_count = 1
        for i in range(1, len(idx)):
            if idx[i] == idx[i-1] + 1:
                counter += 1
                if counter > max_count:
                    max_count = counter
            else:
                counter = 1
        return max_count

    else:

        return len(idx)  # faster than summing over masks


def rand_geom(start, end):
    return np.exp(np.random.uniform(np.log(start), np.log(end)))


def cummin(a):
    return [min(a[:idx+1]) for idx in range(len(a))]


def apply_to_dict(fun, dct):
    dct_new = {}
    for key, val in dct.iteritems():
        dct_new[key] = fun(dct[key])
    return dct_new


def to_precision(x,p):
    """
    Source: http://randlet.com/blog/python-significant-figures-format/
    Changed math.pow to pow...seems to work.
    
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(np.log10(x))
    tens = pow(10, e - p + 1)
    n = np.floor(x/tens)

    if n < pow(10, p - 1):
        e = e -1
        tens = pow(10, e - p+1)
        n = np.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= pow(10,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)


# def expand_array(a, desired_len):
#     scale_factor = desired_len / len(a)
#     assert scale_factor % 1 == 0, 'desired_len needs to be a multiple of the input array'
#     scale_factor = int(scale_factor)
#     expanded = np.empty(desired_len, dtype=type(a[0]))
#     for i in range(scale_factor):
#         expanded[i::scale_factor] = a
#     return expanded


# class StandardScalerColumns:

#     def __init__(self):
#         self.mean = None
#         self.std = None
        
#     def fit(self, Y):
#         self.mean = np.mean(Y, axis=0)
#         self.std = np.std(Y, axis=0)
#         return self
        
#     def transform(self, Y):
#         self.check_if_fitted()
#         return (Y - self.mean) / self.std
    
#     def transform_back(self, Y_scaled):
#         self.check_if_fitted()
#         return Y_scaled * self.std + self.mean
    
#     def check_if_fitted(self):
#         if self.mean is None or self.std is None:
#             raise ValueError('Need to fit scaler first')

