
from __future__ import division

import os
import shutil
import numpy as np
import pandas as pd
import time
from collections import OrderedDict
from datetime import datetime
from datetime import timedelta
from datetime import date

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from utils import *


def load_house_csv(house_id, dir_refit_csv, nrows=None):
    '''
    '''
    csv_filename = os.path.join(dir_refit_csv, 'CLEAN_House{}.csv'.format(house_id))
    df = pd.read_csv(csv_filename, nrows=nrows)

    return df


def save_refit_data(dir_refit_csv, dir_refit_np, nrows=None, house_ids=None):
    '''
    '''

    if house_ids is None:
        house_ids = range(1, 22)
        house_ids.remove(14)  # no house 14

    print 'writing REFIT data...'

    for house_id in house_ids:

        t0 = time.time()

        # Load data for a house.
        df = load_house_csv(house_id, dir_refit_csv, nrows=nrows)

        if house_id == 1:
            # Create directory where we'll save REFIT data.
            if os.path.exists(dir_refit_np):
                shutil.rmtree(dir_refit_np)
            os.makedirs(dir_refit_np)

        # Make directory for house.
        dir_house = os.path.join(dir_refit_np, 'house{}'.format(house_id))
        os.makedirs(dir_house)
        
        columns = list(df)
        columns.remove('Time')  # can just use Unix since Pandas to_datetime gives same value
        for column in columns:
            # Data type should be int32 for timestamp and int16 for appliance power levels.
            dtype = np.int32 if column == 'Unix' else np.int16
            np.save(os.path.join(dir_house, '{}.npy'.format(column)), df[column].values.astype(dtype))

        t1 = time.time()
        print 'added house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)

    print 'done!'


def create_app_dict():
    return OrderedDict([
        ('fridge', {'col': 'Fridge', 'pattern': 'fridge'}),
        ('kettle', {'col': 'Kettle', 'pattern': 'kettle$'}),  # pattern ignores 'Kettle/Toaster'
        ('washing machine', {'col': 'WashingMachine', 'pattern': 'washing *machine'}),
        ('dishwasher', {'col': 'DishWasher', 'pattern': 'dish *washer'}),
        ('microwave', {'col': 'Microwave', 'pattern': '^microwave'})  # pattern ignores 'Combination Microwave'
    ])


def apps_add_cols_from_patterns(app_dict):
    for (app_name, dct) in app_dict.items():
        app_col = dct['col']
        pattern = dct['pattern']
        apps[app_col] = apps['ApplianceOrig_Raw'].str.lower().str.contains(pattern).astype(int)
    return apps


def create_app_funs(apps, app_dict):

    def get_house_app_tuples(app_name, return_pd=False):
        app_col = app_dict[app_name]['col']
        if return_pd:
            return apps.loc[apps[app_col] == 1]
        house_to_app = apps.loc[apps[app_col] == 1][['House', 'ApplianceNum']].values
        return [(h, a) for h, a in house_to_app]

    def get_app_nums(house_id, app_name):
        app_col = app_dict[app_name]['col']
        cond1 = apps[app_col] == 1
        cond2 = apps['House'] == house_id
        app_nums = apps.loc[(cond1) & (cond2)]['ApplianceNum'].values.tolist()
        return app_nums
    
    def get_app_name(house_id, app_num):
        return apps[(apps['House'] == house_id) & (apps['ApplianceNum'] == app_num)]['Appliance'].values[0]

    return (get_house_app_tuples, get_app_nums, get_app_name)


def create_load_funs(dir_refit):

    def load_app(house_id, app_num):
        np_path = 'Aggregate.npy' if app_num == 0 else 'Appliance{}.npy'.format(app_num)
        return np.load(os.path.join(dir_refit,
                                    'house{}'.format(house_id),
                                    np_path))

    def load_ts(house_id):
        return np.load(os.path.join(dir_refit,
                                    'house{}'.format(house_id),
                                    'Unix.npy'))
    
    def load_issues(house_id):
        return np.load(os.path.join(dir_refit,
                                    'house{}'.format(house_id),
                                    'Issues.npy'))
    
    return (load_app, load_ts, load_issues)


def get_ts_idx(ts_series, dt_start, dt_end=None):
    '''
    Get index for time of day without having to convert all timestamps in array to dateitmes.
    '''
    ts_start = dt2ts(dt_start)
    if dt_end is None:
        # Assume one day.
        ts_end = dt2ts(dt_start + timedelta(days=1))
    else:
        ts_end = dt2ts(dt_end)
    ts_series = pd.DataFrame(ts_series).set_index(0).index
    return (ts_series >= ts_start) & (ts_series < ts_end)


def get_df(house_id, use_app_names=False, dt_start=None, dt_end=None, include_issues=False):
    '''
    Plot time series of power data for each appliance, for specified house and date(time).
    '''
    
    # Load time series for house and get timestamps for specified date.
    ts_series = load_ts(house_id)
    
    # Add first column to df (timestamp).
    df = pd.DataFrame({'Unix': ts_series})
    
    # Add appliance columns.
    for app_num in range(10):
        if use_app_names:
            app_name = get_app_name(house_id, app_num)
        else:
            app_name = 'Appliance{}'.format(app_num) if app_num>0 else 'Aggregate'
        df[app_name] = load_app(house_id, app_num)
        
    if dt_start is not None:  # that is, if we don't want all dates
        idx = get_ts_idx(ts_series, dt_start, dt_end)
        df = df.loc[idx]
    
    if include_issues:
        # Add issues column.
        df['Issues'] = load_issues(house_id)
    
    return df


def plot_day(house_id, dt, savefile=None, figsize=(7,5), cols=None):
    '''
    Plot time series of power data for each appliance, for specified house and date(time).
    '''
    df = get_df(house_id, use_app_names=True, dt_start=dt)
    if cols is not None:
        cols += ['Unix']  # add Unix in case it wasn't included in cols
        cols = list(set(cols))
        df = df[cols]
    df['Time'] = pd.to_datetime(df['Unix'], unit='s', utc=True)
    df.set_index('Time', inplace=True)
    del df['Unix']
    
    # df = df.tz_localize('GMT').tz_convert('Europe/London')
    ax = df.plot(figsize=figsize)
    ax.set_title('House {}\n{}'.format(house_id, dt.date().strftime('%Y-%m-%d')))
    ax.set_xlabel('')
    # plt.xticks(np.arange(min(df.index), max(df.index)+1, 8.))
    if savefile is not None:
        plt.savefig(savefile)
    return ax


def plot_date_range(house_id, dt_base, day_range=range(-7, 5), app_names=None, figsize=(11,2)):
    if isinstance(app_names, basestring):
        app_names = [app_names]  # if they were entered as string, convert to list for Pandas
    for day_delta in day_range:
        dt = dt_base + timedelta(days=day_delta)
        try:
            plot_day(house_id, dt, figsize=figsize, cols=app_names)
        except TypeError:
            print 'no data for {}'.format(str(dt.date()))


def calc_stats_for_house(house_id, nrow=None):
    '''
    Calculate daily summary stats for each home and some appliance in some home.
    '''
    
    WATTSEC_2_KWH = 1/3.6e6  # multiply by this to convert from Watt-sec to kWh

    # Define functions to be used in aggregation.
    funs = OrderedDict([
        ('RowNum', len),
        ('Time', [min, max]),
        ('UnixDiff', max),
        ('EnergySumOfParts', sum),
        # ('CorrCoef', lambda x: x.max()),
        ('PctAccountedEnergy', np.std),
        ('Issues', [sum, np.mean])
    ])
    for app_num in range(10):
        # Calculate total energy used by appliance.
        funs['EnergyAppliance{}'.format(app_num)] = sum
    
    # Import timestamp data, calculate diffs (for calculating energy used by appliances), and set index (for grouping by day)
    ts_series = load_ts(house_id)
    df = pd.DataFrame({'Unix': ts_series})  # will convert to Time later
    df['Issues'] = load_issues(house_id)
    if nrow is not None:
        df = df.iloc[range(nrow)]
    df['UnixDiff'] = df['Unix']  # will take diff later in grouping
    df['RowNum'] = range(df.shape[0])
    df['Time'] = pd.to_datetime(df['Unix'], unit='s', utc=True)
    df.set_index('Time', inplace=True)
    # df = df.tz_localize('GMT').tz_convert(tz)
    df = df.groupby(pd.TimeGrouper(freq='D')).transform(lambda x: calc_diff(x, True, 6))
    df['Time'] = df.index  # need this twice: one for grouping and one for calculating stats
    
    # Calculate energy used by appliances.
    df['EnergySumOfParts'] = np.zeros(df.shape[0], dtype=np.int)
    for app_num in range(10):
        app_data = load_app(house_id, app_num)
        if nrow is not None:
            app_data = app_data[range(nrow)]
        df['Appliance{}'.format(app_num)] = app_data
        df['EnergyAppliance{}'.format(app_num)] = df['Appliance{}'.format(app_num)] * df['UnixDiff'] * WATTSEC_2_KWH
        if app_num > 0:
            df['EnergySumOfParts'] += df['EnergyAppliance{}'.format(app_num)]
    df['PctAccountedEnergy'] = df['EnergySumOfParts'] / df['EnergyAppliance0']
    
    # Calculation correlation between sum of appliances and main.
    corr = df[['EnergySumOfParts', 'EnergyAppliance0']].groupby(pd.TimeGrouper(freq='D')).corr().ix[0::2,'EnergyAppliance0']
    corr = corr.reset_index().drop('level_1', axis=1)
    corr.set_index('Time', inplace=True)
    
    # Aggregate by day and apply functions.
    dstats = df.groupby(pd.TimeGrouper(freq='D')).aggregate(funs)
    assert corr.shape[0] == dstats.shape[0]
    dstats['SumToMainCorr'] = corr['EnergyAppliance0']
    
    # Calculate percent of aggregate energy accounted for by appliances.
    dstats[('PctAccountedEnergy', 'total')] = dstats[('EnergySumOfParts', 'sum')] / dstats[('EnergyAppliance0', 'sum')]
    
    # Get timestamp range in hours.
    dstats['HourRange'] = dstats[('Time', 'max')] - dstats[('Time', 'min')]
    dstats['HourRange'] = dstats['HourRange'] / np.timedelta64(1, 'h')
    
    # Make TimestampDiff in hours.
    dstats[('UnixDiff', 'max')] = dstats[('UnixDiff', 'max')]/60/60

    # Calculate number of hours in day (!= 24, e.g., for daylight savings switchover days).
    dstats['HoursInDay'] = [hours_in_day(dt) for dt in dstats.index]

    return dstats


def create_daily_stats(house_ids, pkl_path=None, nrow=None):

    dstats = []
    for house_id in house_ids:
        print 'calculating stats for house {}...'.format(house_id)
        stats_house = calc_stats_for_house(house_id, nrow=nrow)
        stats_house['House'] = house_id
        dstats.append(stats_house)
    dstats = pd.concat(dstats)
    print 'done!'
    
    if pkl_path is not None:
        dstats.to_pickle(pkl_path)
    
    return dstats


def create_daily_plots(house_ids, dir_run):

    # Plot day of data for all homes for various days (approx one plot per month).
    dt_start = datetime(2013,11,1)  # around first day of data
    for house_id in house_ids:

        print 'making plots for house {}...'.format(house_id)

        house_dir = os.path.join(dir_run, 'daily_power', 'house{}'.format(house_id))
        if not os.path.exists(house_dir):
            os.makedirs(house_dir)

        # Iterate over num months to add to starting datetime.
        for days_to_add in np.array(range(20))*31:
            dt = dt_start + timedelta(days=days_to_add)
            savefile = os.path.join(house_dir, '{}.pdf'.format(str(dt.date())))
            try:
                plot_day(house_id, dt, savefile)
            except TypeError:
                # DataFrame has no columns b/c there's no data for that day.
                pass
                # print 'No data to plot for house{}_{}.pdf'.format(house_id, str(dt.date()))
            plt.close()


def get_energy(dstats, house_id, dt, app_nums):
    '''
    When there are multiple cols for an appliance, it adds them.
    '''
    if not app_nums:
        # If home doesn't have this appliance, return zero energy.
        return 0.
    
    # Subset by house and appliance.
    cols = ['EnergyAppliance{}'.format(s) for s in app_nums]
    energy = dstats.loc[(dstats['House'] == house_id)][cols]
    
    # Subset by datetime.
    energy = energy.loc[str(dt.date())]
    
    return sum(energy.values)


def clean_daily_stats(dstats):

    print 'cleaning daily stats...'

    conds = [
        dstats[('RowNum', 'len')] < 500,
        dstats[('UnixDiff', 'max')] > 0.25,  # 15 min
        dstats['HourRange'] < 23.5,
        # dstats[('PctAccountedEnergy', 'identity')] > float('inf'),
        # dstats[('PctAccountedEnergy', 'std')] > float('inf'),
        # dstats['SumToMainCorr'] < 0.25,
        dstats[('Issues', 'mean')] > 0.05,
        dstats['HoursInDay'] != 24,
        dstats['House'].isin([1, 11, 21]),
        dstats[('EnergyAppliance0', 'sum')] < 0
    ]
    total_rows = dstats.shape[0]
    delete = pd.DataFrame({'Timestamp': dstats.index,
                           'Delete': np.zeros(total_rows, dtype=int)})
    delete.set_index('Timestamp', inplace=True)
    for cond_num, cond in enumerate(conds):
        array = cond.as_matrix()
        rows_affected = sum(array)
        new_rows_deleted = sum([i == 0 and j == 1 for i, j in zip(delete['Delete'].values, array)])
        delete.loc[array] = 1
        print 'Condition {}: {} ({:0.2g}%) rows affected in total, {} ({:0.2g}%) new'.format(
            cond_num+1,
            rows_affected,
            rows_affected / total_rows * 100,
            new_rows_deleted,
            new_rows_deleted / total_rows * 100
        )
    rows_deleted = sum(delete['Delete'].values)
    print 'Total rows marked for deletion: {} ({:0.2g}%)'.format(rows_deleted, rows_deleted / total_rows * 100)
    
    dstats['Delete'] = delete['Delete']
    
    return dstats


def create_data(house_ids, app_names, dstats, dir_data=None, desired_sample_rate=6, is_debug=True):
    
    X = []
    Y = []
    x_house = []
    x_date = []
    
    for house_id in house_ids:
        
        if is_debug:
            print 'Creating data for house {}...'.format(house_id)
            t0 = time.time()

        app_name2nums = OrderedDict()
        for app_name in app_names:
            app_name2nums[app_name] = get_app_nums(house_id, app_name)

        ts_series = load_ts(house_id)
        main = load_app(house_id, 0)

        dt_min = floor_time(ts2dt(int(ts_series.min())))
        dt_max = floor_time(ts2dt(int(ts_series.max())))

        # Account for weird issue where max date in numpy array is less than max date in
        # aggregate stats dataframe. Notably, for house 8, datetime(2015,5,11)
        # exists in the array but not in the dataframe.
        dt_min_dstats = dstats.loc[dstats['House']==house_id].index.min()
        dt_max_dstats = dstats.loc[dstats['House']==house_id].index.max()
        if dt_min != dt_min_dstats or dt_max != dt_max_dstats:
            print 'warning: fixing wrong min date ({} vs {}) or max date ({} vs {})'.format(
                dt_min.date(),
                dt_min_dstats.date(),
                dt_max.date(),
                dt_max_dstats.date())
            dt_min = max(dt_min, dt_min_dstats)
            dt_max = min(dt_max, dt_max_dstats)

        all_dts = [dt_min + timedelta(days=d) for d in range(0, (dt_max-dt_min).days+1)]

        for dt_start in all_dts:

            # For debugging that weird day that throws an error for house 8.
            # if dt_start != datetime(2015,5,11) or house_id != 8:
            #     continue

            # Check if we want to process this data.
            data_okay = dstats.loc[dstats['House']==house_id].loc[str(dt_start.date())]['Delete'].values[0] == 0
            if not data_okay:
                continue

            dt_end = dt_start + timedelta(days=1)
            ts_idx = get_ts_idx(ts_series=ts_series,
                                dt_start=dt_start)
            ts_day_actual = ts_series[ts_idx]
            ts_day_desired = range(dt2ts(dt_start) + desired_sample_rate,
                                   dt2ts(dt_end) + desired_sample_rate,
                                   desired_sample_rate)
                
            aligned_idx = align_arrays(ts_day_actual, ts_day_desired, padder=0)

            x = main[ts_idx][aligned_idx]
            y = []
            for app_name in app_names:
                y.append(get_energy(dstats, house_id, dt_start, app_name2nums[app_name]))

            X.append(x)
            x_house.append(house_id)
            x_date.append(dt_start.date())
            Y.append(y)

        if is_debug:
            t1 = time.time()
            print 'Created data for house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)
    
    if dir_data is not None:
        np.save(os.path.join(dir_data, 'X.npy'), X)
        np.save(os.path.join(dir_data, 'Y.npy'), Y)
        np.save(os.path.join(dir_data, 'x_house.npy'), x_house)
        np.save(os.path.join(dir_data, 'x_date.npy'), x_date)
    
    return X, Y, x_house, x_date


if __name__ == '__main__':

    dir_proj = '/Users/sipola/Google Drive/education/coursework/graduate/edinburgh/dissertation/thesis'
    dir_data = os.path.join(dir_proj, 'data')
    dir_run = os.path.join(dir_proj, 'run', str(date.today()))

    dir_refit_csv = os.path.join(dir_data, 'CLEAN_REFIT_081116')
    dir_refit = os.path.join(dir_data, 'refit')
    
    path_apps = os.path.join(dir_data, 'appliances.csv')
    path_daily_stats = os.path.join(dir_data, 'stats_by_day.pkl')

    HOUSE_IDS = range(1, 22); HOUSE_IDS.remove(14)  # no house 14
    APP_NAMES = ['fridge', 'kettle', 'washing machine', 'dishwasher', 'microwave']

    # save_refit_data(dir_refit_csv=dir_refit_csv, dir_refit_np=dir_refit, nrows=None)

    apps = pd.read_csv(path_apps)
    app_dict = create_app_dict()
    apps = apps_add_cols_from_patterns(app_dict)

    get_house_app_tuples, get_app_nums, get_app_name = create_app_funs(apps, app_dict)
    load_app, load_ts, load_issues = create_load_funs(dir_refit)

    # create_daily_plots(HOUSE_IDS, dir_run)

    # dstats = create_daily_stats(HOUSE_IDS, pkl_path=path_daily_stats, nrow=None)
    dstats = pd.read_pickle(path_daily_stats)
    dstats = clean_daily_stats(dstats)

    X, Y, x_house, x_date = create_data(HOUSE_IDS, APP_NAMES, dstats, dir_data)