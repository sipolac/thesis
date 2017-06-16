
import numpy as np
from datetime import datetime
from datetime import timedelta


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


def floor_time(dt):
    return datetime.combine(dt.date(), datetime.min.time())


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


# class StdevFunc:
#     '''
#     A standard dev function for SQLite
#     '''
#     # Credit: http://www.alexforencich.com/wiki/en/scripts/python/stdev
#     def __init__(self):
#         self.M = 0.0
#         self.S = 0.0
#         self.k = 1
 
#     def step(self, value):
#         if value is None:
#             return
#         tM = self.M
#         self.M += (value - tM) / self.k
#         self.S += (value - tM) * (value - self.M)
#         self.k += 1
 
#     def finalize(self):
#         if self.k < 3:
#             return None
#         return np.sqrt(self.S / (self.k-2))