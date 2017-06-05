
import numpy as np
import datetime


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ts2dt(ts):
    '''Convert int timestamp to datetime'''
    # https://stackoverflow.com/questions/3694487/initialize-a-datetime-object-with-seconds-since-epoch
    # return np.array(pd.to_datetime(ts, unit='s', utc=True).tz_convert('Europe/London'))
    if isinstance(ts, int):
        return datetime.datetime.fromtimestamp(ts)
    else:
        return [ts2dt(i) for i in ts]


def dt2ts(dt):
    '''Convert datetime to int timestamp'''
    # https://stackoverflow.com/questions/11743019/convert-python-datetime-to-epoch-with-strftime
    if isinstance(dt, datetime.datetime):
        return int(dt.strftime('%s'))
    else:
        return [dt2ts(i) for i in dt]


def test_ts_dt_conversions(v1=1381323977):
    v2 = dt2ts(ts2dt(v1))
    if v1 == v2:
        return 'Values are the same'
    else:
        return 'Values are different: started as {} and ended as {}'.format(ts2dt(v1), ts2dt(v2))