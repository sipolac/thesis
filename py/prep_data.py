'''
TODO:
> Fix issue with "app_name" referring to either standardized (e.g., "washing machine",
"fridge") and non-standardized (e.g., "washing machine (1)", "fridge-freezer"). Don't
want to make mistake of searching for a non-standardized names in a list of standardized
names...you will miss some.
'''

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

from utils import *

matplotlib.style.use('ggplot')


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
        print 'wrote data for house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)

    print 'done!'


def create_app_dict():
    return OrderedDict([
        ('fridge', {'col': 'Fridge', 'pattern': 'fridge'}),
        ('kettle', {'col': 'Kettle', 'pattern': 'kettle$'}),  # pattern ignores 'Kettle/Toaster'
        ('washing machine', {'col': 'WashingMachine', 'pattern': 'washing *machine'}),
        ('dishwasher', {'col': 'DishWasher', 'pattern': 'dish *washer'}),
        ('microwave', {'col': 'Microwave', 'pattern': '^microwave'})  # pattern ignores 'Combination Microwave'
    ])


def apps_add_cols_from_patterns(apps, app_dict):
    for (app_name, dct) in app_dict.items():
        app_col = dct['col']
        pattern = dct['pattern']
        apps[app_col] = apps['ApplianceOrig_Raw'].str.lower().str.contains(pattern).astype(int)
    return apps


def create_app_funs(apps, app_dict, app_names):

    app_cols = [app_dict[app_name]['col'] for app_name in app_names]

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

    def is_a_target_app(house_id, app_num):
        return (apps.loc[(apps['House']==house_id) & (apps['ApplianceNum']==app_num)][app_cols].values).any()

    return (get_house_app_tuples, get_app_nums, get_app_name, is_a_target_app)


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


def get_ts_mask(ts_series, dt_start, dt_end=None):
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
        ts_mask = get_ts_mask(ts_series, dt_start, dt_end)
        df = df.loc[ts_mask]
    
    if include_issues:
        # Add issues column.
        df['Issues'] = load_issues(house_id)
    
    return df


def plot_day(house_id, dt, savefile=None, figsize=(9,5), cols=None):
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    app_names = list(df)  # get columns from DataFrame
   
    # Set color map.
    colormap = plt.cm.Set1
    ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, len(app_names))])
    # ax = df.plot(figsize=figsize)
    for app_name in app_names:
        ax = df[app_name].plot(figsize=figsize)
    ax.set_title('House {}\n{}'.format(house_id, dt.date().strftime('%Y-%m-%d')))
    ax.set_xlabel('')
    ax.set_ylabel('Power (Watts)')
    # plt.xticks(np.arange(min(df.index), max(df.index)+1, 8.))

    # Put legend outside of plot.
    # https://stackoverflow.com/a/4701285/4794432
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # Decrese legend font size.
    # fontP = FontProperties()
    # fontP.set_size('xx-small')

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
        funs['Appliance{}'.format(app_num)] = [np.mean, np.median, min, max]
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

    print 'calculating daily stats...'

    dstats = []
    for house_id in house_ids:
        t0 = time.time()
        print 'calculating stats for house {}...'.format(house_id)
        stats_house = calc_stats_for_house(house_id, nrow=nrow)
        stats_house['House'] = house_id
        dstats.append(stats_house)
        t1 = time.time()
        print 'calculated stats for house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)
    dstats = pd.concat(dstats)
    print 'done!'
    
    if pkl_path is not None:
        dstats.to_pickle(pkl_path)
    
    return dstats


def create_daily_plots(house_ids, dir_run, figsize=(9,5)):

    # Plot day of data for all homes for various days (approx one plot per month).

    print 'creating daily plots...'

    dt_start = datetime(2013,11,1)  # around first day of data
    for house_id in house_ids:

        print 'creating plots for house {}...'.format(house_id)
        t0 = time.time()

        house_dir = os.path.join(dir_run, 'daily_power', 'house{}'.format(house_id))
        if not os.path.exists(house_dir):
            os.makedirs(house_dir)

        # Iterate over num months to add to starting datetime.
        for days_to_add in np.array(range(20))*31:
            dt = dt_start + timedelta(days=days_to_add)
            savefile = os.path.join(house_dir, '{}.pdf'.format(str(dt.date())))
            try:
                plot_day(house_id, dt, savefile, figsize=figsize)
            except TypeError:
                # DataFrame has no columns b/c there's no data for that day.
                pass
                # print 'No data to plot for house{}_{}.pdf'.format(house_id, str(dt.date()))
            plt.close()

        t1 = time.time()
        print 'created plots for house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)


def get_energy(dstats, house_id, d, app_nums):
    '''
    When there are multiple cols for an appliance, it adds them.
    '''
    if not app_nums:
        # If home doesn't have this appliance, return zero energy.
        return 0.

    # Subset by house and appliance.
    cols = [('EnergyAppliance{}'.format(s), 'sum') for s in app_nums]
    energy = dstats.loc[(dstats['House'] == house_id)][cols]
    
    # Subset by datetime.
    energy = energy.loc[str(d)]
    
    return sum(energy.values)


def clean_daily_stats(dstats):

    print 'cleaning daily stats...'

    conds = OrderedDict([
        (('RowNum', 'len'), dstats[('RowNum', 'len')] < 500),
        (('UnixDiff', 'max'), dstats[('UnixDiff', 'max')] > 0.25),  # 15 min
        ('HourRange', dstats['HourRange'] < 23.5),
        (('Issues', 'mean'), dstats[('Issues', 'mean')] > 0.05),
        ('HoursInDay', dstats['HoursInDay'] != 24),
        ('House', dstats['House'].isin([3,11,21]))  # solar panels
    ])
    for app_num in range(10):
        # Calculate total energy used by appliance.
        col = ('Appliance{}'.format(app_num), 'mean')
        conds[col] = dstats[col] < 0

        col = ('Appliance{}'.format(app_num), 'median')
        conds[col] = dstats[col] < 0

        col = ('Appliance{}'.format(app_num), 'min')
        conds[col] = dstats[col] < 0

        col = ('EnergyAppliance{}'.format(app_num), 'sum')
        conds[col] = dstats[col] < 0

    total_rows = dstats.shape[0]
    delete = pd.DataFrame({'Timestamp': dstats.index,
                           'Delete': np.zeros(total_rows, dtype=int)})
    delete.set_index('Timestamp', inplace=True)
    for cond_num, (col, cond) in enumerate(conds.iteritems()):
        array = cond.as_matrix()
        rows_affected = sum(array)
        new_rows_deleted = sum([i == 0 and j == 1 for i, j in zip(delete['Delete'].values, array)])
        delete.loc[array] = 1
        print '{} ({:0.2g}%) rows affected in total, {} ({:0.2g}%) new | Condition {} ({})'.format(
            rows_affected,
            rows_affected / total_rows * 100,
            new_rows_deleted,
            new_rows_deleted / total_rows * 100,
            cond_num+1,
            str(col)
        )
    rows_deleted = sum(delete['Delete'].values)
    print 'Total rows marked for deletion: {} ({:0.2g}%)'.format(rows_deleted, rows_deleted / total_rows * 100)
    
    dstats['Delete'] = delete['Delete']
    
    return dstats


def data_okay(house_id, d):
    return dstats.loc[dstats['House']==house_id].loc[str(d)]['Delete'].values[0] == 0


def get_all_dts_for_house(house_id, train_dts=None):
    
    ts_series = load_ts(house_id)
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
    if train_dts is not None:
        all_dts = [dt for dt in all_dts if dt in train_dts]  # only in given range
        
    return all_dts


def get_aligned_ts_mask_for_day(ts_series, dt_start, desired_sample_rate):
    
    dt_end = dt_start + timedelta(days=1)
    ts_mask = get_ts_mask(ts_series=ts_series,
                        dt_start=dt_start)
    ts_day_actual = ts_series[ts_mask]
    ts_day_desired = range(dt2ts(dt_start) + desired_sample_rate,
                           dt2ts(dt_end) + desired_sample_rate,
                           desired_sample_rate)

    return (align_arrays(ts_day_actual, ts_day_desired, padder=0), ts_mask)


def create_real_data(house_ids, app_names, dstats, desired_sample_rate, save_dir=None, is_debug=True):
    
    
    if is_debug:
        print 'creating real data...'

    X = []
    Y = []
    x_house = []
    x_date = []
    
    # OVERWERITE
    if save_dir is not None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    for house_id in house_ids:
        
        if is_debug:
            print 'creating real data for house {}...'.format(house_id)
            t0 = time.time()

        app_name2nums = OrderedDict()
        for app_name in app_names:
            app_name2nums[app_name] = get_app_nums(house_id, app_name)

        ts_series = load_ts(house_id)
        main = load_app(house_id, 0)
        
        all_dts = get_all_dts_for_house(house_id)

        for dt_start in all_dts:

            # For debugging that weird day that throws an error for house 8.
            # if dt_start != datetime(2015,5,11) or house_id != 8:
            #     continue

            # Check if we want to process this data.
            if not data_okay(house_id, dt_start.date()):
                continue

            aligned_idx, ts_mask = get_aligned_ts_mask_for_day(ts_series, dt_start, desired_sample_rate)

            x = main[ts_mask][aligned_idx]
            y = []
            for app_name in app_names:
                y.append(get_energy(dstats, house_id, dt_start.date(), app_name2nums[app_name]))

            X.append(x)
            x_house.append(house_id)
            x_date.append(dt_start.date())
            Y.append(y)

        if is_debug:
            t1 = time.time()
            print 'created real data for house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)
    
    if save_dir is not None:
        np.save(os.path.join(save_dir, 'X.npy'), X)
        np.save(os.path.join(save_dir, 'Y.npy'), Y)
        np.save(os.path.join(save_dir, 'x_house.npy'), x_house)
        np.save(os.path.join(save_dir, 'x_date.npy'), x_date)
    
    return X, Y, x_house, x_date


def create_bank_choices(dstats, app_names, house_ids, train_dts):

    '''
    Only includes target appliances.
    '''

    # Subset stats df
    df = dstats.copy()
    df = df.loc[df['Delete']==0]
    df = df.loc[df['House'].isin(house_ids)]
    df = df.loc[df.index.isin(train_dts)]
    
    df = df[['House']]
    
    bank_choices = []
    for app_name in app_names:
        for house_id, app_num in get_house_app_tuples(app_name):
            one_combo = df.loc[df['House']==house_id].copy()
            one_combo['ApplianceNum'] = app_num
            one_combo['Appliance'] = app_name  # this is the "standardized" app name
            bank_choices.append(one_combo)
    bank_choices = pd.concat(bank_choices)
    
    return bank_choices


def get_random_series_metadata(bank_choices, app_name=None):
    
    bc = bank_choices.copy()
    
    if app_name is not None:
        bc = bc.loc[bc['Appliance']==app_name]
    
    row = bc.sample(1)
    dt = dt64_to_datetime(row.index.values[0]).date()
    house_id = row['House'].values[0]
    app_num = row['ApplianceNum'].values[0]
    
    return [house_id, app_num, dt]


def get_aligned_series(house_id, app_num, dt, desired_sample_rate=6):
    ts_series = load_ts(house_id)
    if isinstance(dt, date):
        dt = date_to_datetime(dt)
    aligned_idx, ts_mask = get_aligned_ts_mask_for_day(ts_series, dt, desired_sample_rate)
    app_series = load_app(house_id, app_num)
    x = app_series[ts_mask][aligned_idx]
    return x

# def get_random_aligned_series_and_energy(app_name):
#     return get_aligned_series(*get_random_series_metadata(app_name))


# def create_bank_of_power_series(app_names, dir_data, desired_sample_rate):

#     print 'creating bank of power series for appliances of interest...'

#     X = []
#     x_house = []
#     x_date = []

#     dir_bank = os.path.join(dir_data, 'bank')

#     for app_name in app_names:

#         print 'creating power series for {}...'.format(app_name)

#         dir_bank_app = os.path.join(dir_bank, app_name)
#         if os.path.exists(dir_bank_app):
#             shutil.rmtree(dir_bank_app)
#         os.makedirs(dir_bank_app)

#         for house_id, app_num in get_house_app_tuples(app_name):

#             print '    house {}, appliance {}...'.format(house_id, app_num)

#             ts_series = load_ts(house_id)
#             all_dts = get_all_dts_for_house(house_id)

#             app_series = load_app(house_id, app_num)

#             for dt_start in all_dts:

#                 if not data_okay(house_id, dt_start.date()):
#                     continue

#                 aligned_idx, ts_mask = get_aligned_ts_mask_for_day(ts_series, dt_start, desired_sample_rate)
#                 x = app_series[ts_mask][aligned_idx]

#                 X.append(x)
#                 x_house.append(house_id)
#                 x_date.append(dt_start.date())

#         # Save data for appliance.
#         filename_and_np_array = [('power.npy', X),
#                                  ('house.npy', x_house),
#                                  ('date.npy', x_date)]
#         for filename, np_array in filename_and_np_array:
#             np.save(os.path.join(dir_bank_app, filename), np_array)
                
#     return X, x_house, x_date


def create_synthetic_data(
    dstats,
    house_ids,
    train_dts,
    app_names,
    swap_prob,
    include_distractor_prob,
    save_dir=None,
    is_debug=False
    ):

    desired_sample_rate = 6
    bank_choices = create_bank_choices(dstats, app_names, house_ids, train_dts)
    # bank_choices = bank_choices.loc[bank_choices['IsTarget']==1]  # just want target apps

    print 'creating synthetic data...'

    X = []
    Y = []
    x_house = []
    x_date = []

    if save_dir is not None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    for house_id in house_ids:

        print 'creating synthetic data based on house {}'.format(house_id)

        ts_series = load_ts(house_id)
        
        all_dts = get_all_dts_for_house(house_id, train_dts)

        for dt_start in all_dts:

            print '    {}'.format(dt_start.date())

            # Check if we want to process this data.
            if not data_okay(house_id, dt_start):
                continue

            aligned_idx, ts_mask = get_aligned_ts_mask_for_day(ts_series, dt_start, desired_sample_rate)

            # Initialize aggregate signal and energy of target appliances.
            x = np.zeros(int(24 * 60 * 60 / desired_sample_rate), dtype=np.int16)
            y = []

            if is_debug:
                print '        adding target appliance signals...'

            # Iterate over target appliances first so we have 
            for app_name in app_names:

                app_nums = get_app_nums(house_id, app_name)

                if is_debug:
                    print '        app_name: {}, app_nums: {}'.format(app_name, app_nums)

                if not app_nums:
                    y.append(0.)
                    if is_debug:
                        print '        skipping (adding all zeros for x and y) since there`s no app'
                    continue

                x_app = np.zeros(int(24 * 60 * 60 / desired_sample_rate), dtype=np.int16)
                y_app = 0.
                for app_num in app_nums:

                    if is_debug:
                        print '        app_name: {}, app_num: {}'.format(app_name, app_num)

                    # If the swap triggers, choose the appliance signal at random from homes that have that appliance.
                    if np.random.rand() < swap_prob:
                        house_id_rand, app_num_rand, d_rand = get_random_series_metadata(bank_choices, app_name)
                        x_app += get_aligned_series(house_id_rand, app_num_rand, d_rand)
                        y_app += get_energy(dstats, house_id_rand, d_rand, [app_num_rand])

                        if is_debug:
                            print '        swapping! random series: house {}, app num {}, date {}'.format(house_id_rand, app_num_rand, d_rand)
                            print '        x_app: {}...'.format(x_app[:5])
                            print '        y_app: {}'.format(y_app)

                    # Otherwise, use actual data from home.
                    else:
                        x_app += load_app(house_id, app_num)[ts_mask][aligned_idx]
                        y_app += get_energy(dstats, house_id, dt_start.date(), [app_num])

                        if is_debug:
                            print '        not swapping'
                            print '        x_app: {}...'.format(x_app[:5])
                            print '        y_app: {}'.format(y_app)

                # Add target appliance signal to synthetic aggregate.
                x += x_app
                y.append(y_app)

            # Add distractor appliances by iterating through appliances in home.
            if is_debug:
                print '        adding distractor appliance signals...'

            for app_num in range(1, 10):  # don't include aggregate

                if is_debug:
                    print '        app_name: {}; app_num: {}'.format(get_app_name(house_id, app_num), app_num)

                # If appliance *is* a target appliance or distractor prob *doesn't*
                # trigger, then skip appliance.
                if is_a_target_app(house_id, app_num) or not np.random.rand() < include_distractor_prob:
                    if is_debug:
                        print '        is either target app or distractor prob doesn`t trigger' 
                    continue
                
                # Add distractor signal to synthetic aggregate.
                x_app = load_app(house_id, app_num)[ts_mask][aligned_idx]
                x += x_app

                if is_debug:
                    print '        added distractor app'
                    print '        x_app: {}...'.format(x_app[:5])

            if is_debug:
                print '        agggregated x: {}...'.format(x[:5])
                print '        y`s: {}'.format(y)

            X.append(x)
            x_house.append(house_id)
            x_date.append(dt_start.date())
            Y.append(y)

        # Save data after every home, just in case something crashes.
        # Later, when code is good, it'd be best to do this just once
        # so you don't overwrite the prior files.
        if save_dir is not None:
            np.save(os.path.join(save_dir, 'X.npy'), X)
            np.save(os.path.join(save_dir, 'Y.npy'), Y)
            np.save(os.path.join(save_dir, 'x_house.npy'), x_house)
            np.save(os.path.join(save_dir, 'x_date.npy'), x_date)

    return X, Y, x_house, x_date


def get_num_runs(dir_for_model_synth):
    '''
    Gets number of times the synthetic data was run
    '''
    num_runs = 0
    for filename in os.listdir(dir_for_model_synth):
        try:
            filename_int = int(filename)
            num_runs = max(num_runs, filename_int)
        except ValueError:
            pass
    return num_runs



def load_real_data(dir_for_model_real):

    X = np.load(os.path.join(dir_for_model_real, 'X.npy'))
    Y = np.load(os.path.join(dir_for_model_real, 'Y.npy'))
    x_house = np.load(os.path.join(dir_for_model_real, 'x_house.npy'))
    x_date = np.load(os.path.join(dir_for_model_real, 'x_date.npy'))

    return X, Y, x_house, x_date


def load_synth_data(dir_for_model_synth, save=False):

    num_runs = get_num_runs(dir_for_model_synth)

    X = []
    Y = []
    x_house = []
    x_date = []
    for run_num in range(1,num_runs+1):

        dir_run_num = os.path.join(dir_for_model_synth, '{}'.format(run_num))

        X_run = np.load(os.path.join(dir_run_num, 'X.npy'))
        Y_run = np.load(os.path.join(dir_run_num, 'Y.npy'))
        x_house_run = np.load(os.path.join(dir_run_num, 'x_house.npy'))
        x_date_run = np.load(os.path.join(dir_run_num, 'x_date.npy'))

        X.append(X_run)
        Y.append(Y_run)
        x_house.append(x_house_run)
        x_date.append(x_date_run)

    X = np.concatenate(X)  # cuts down to correct number of obsw
    Y = np.concatenate(Y)
    x_house = np.concatenate(x_house)
    x_date = np.concatenate(x_date)

    if save:
        np.save(os.path.join(dir_for_model_synth, 'X.npy'), X)
        np.save(os.path.join(dir_for_model_synth, 'Y.npy'), Y)
        np.save(os.path.join(dir_for_model_synth, 'x_house.npy'), x_house)
        np.save(os.path.join(dir_for_model_synth, 'x_date.npy'), x_date)
    
    return X, Y, x_house, x_date


def get_train_dts(dstats, prop_train, save_dir=None, is_debug=False):

    dstats_good = dstats.loc[dstats['Delete']==0]
    good_days = dstats_good.index.values
    good_days = np.array(list(set(good_days)))
    train_dts = np.random.choice(good_days, size=int(len(good_days)*prop_train), replace=False)  # in np64 dtype
    train_dts.sort()
    if is_debug:
        pct_train = dstats_good[dstats_good.index.isin(train_dts)].shape[0] / dstats_good.shape[0]
        print '{:0.2g}% training data'.format(pct_train*100)

    train_dts = [dt64_to_datetime(dt) for dt in train_dts]  # convert to datetime
    if save_dir is not None:
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
        np.save(os.path.join(save_dir, 'train_dts.npy'), train_dts)

    return train_dts


if __name__ == '__main__':

    desired_sample_rate = 6  # series created will have timestamps that are this many seconds apart
    swap_prob = 1/2
    include_distractor_prob = 1/2
    synthetic_data_runs = 10
    prop_train = 0.85

    dir_proj = '/Users/sipola/Google Drive/education/coursework/graduate/edinburgh/dissertation/thesis'
    dir_data = os.path.join(dir_proj, 'data')
    dir_run = os.path.join(dir_proj, 'run', str(date.today()))
    dir_run_synthetic = os.path.join(dir_run, 'synthetic')
    dir_run_real = os.path.join(dir_run, 'real')

    dir_refit_csv = os.path.join(dir_data, 'CLEAN_REFIT_081116')
    dir_refit = os.path.join(dir_data, 'refit')
    
    path_apps = os.path.join(dir_data, 'appliances.csv')
    path_daily_stats = os.path.join(dir_data, 'stats_by_day.pkl')

    HOUSE_IDS = range(1, 22); HOUSE_IDS.remove(14)  # no house 14
    APP_NAMES = ['fridge', 'kettle', 'washing machine', 'dishwasher', 'microwave']
    HOUSE_IDS_TEST = [2,9,20]
    HOUSE_IDS_TRAIN_VAL = [house_id for house_id in HOUSE_IDS if house_id not in HOUSE_IDS_TEST]
    TRAIN_VAL_DATE_MAX = datetime(2015,2,28)

    # save_refit_data(dir_refit_csv=dir_refit_csv, dir_refit_np=dir_refit, nrows=None)

    apps = pd.read_csv(path_apps)
    app_dict = create_app_dict()
    apps = apps_add_cols_from_patterns(apps, app_dict)

    get_house_app_tuples, get_app_nums, get_app_name, is_a_target_app = create_app_funs(apps, app_dict, APP_NAMES)
    load_app, load_ts, load_issues = create_load_funs(dir_refit)

    # create_daily_plots(HOUSE_IDS, dir_run)

    # dstats = create_daily_stats(HOUSE_IDS, pkl_path=path_daily_stats, nrow=None)
    dstats = pd.read_pickle(path_daily_stats)
    dstats = clean_daily_stats(dstats)
    train_dts = get_train_dts(dstats, prop_train, save_dir=dir_run_synthetic)

    # X, Y, x_house, x_date = create_real_data(HOUSE_IDS, APP_NAMES, dstats, desired_sample_rate, dir_run_real)

    for run_num in range(1,synthetic_data_runs+1):
        print '=============== RUN NUM {} ==============='.format(run_num)
        X, Y, x_house, x_date = create_synthetic_data(
            dstats,
            HOUSE_IDS_TRAIN_VAL,
            train_dts,
            APP_NAMES,
            swap_prob,
            include_distractor_prob,
            save_dir=os.path.join(dir_run_synthetic, str(run_num)),
            is_debug=False
            )

        '''
    Possible test appliances.
    2. fridge-freezer; standard otherwise
    5. fridge-freezer; has tumble dryer
    9. fridge-freezer; washer-dryer and washing machine
    15. fridge-freezer; has tumble dryer
    20. fridge and freezer; has tumble dryer

    Bad test appliances (missing crucial target appliance or has bad data (solar)).
    1: no kettle
    4. no dishwasher
    3. solar
    6. no fridge-freezer
    8. no dishwasher
    7. no microwave
    10. no kettle
    11. solar
    12. no dishwasher
    13. no fridge/freezer
    16. no kettle
    17. no dishwasher
    18. no kettle
    19. no dishwasher
    21. solar
    '''