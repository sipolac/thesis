'''
TODO:
> Fix issue with "app_name" referring to either standardized (e.g., "washing machine",
"fridge") and non-standardized (e.g., "washing machine (1)", "fridge-freezer"). Don't
want to make mistake of searching for a non-standardized names in a list of standardized
names...you will miss some.
'''

from __future__ import division

from utils import *

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

matplotlib.style.use('ggplot')


# def get_params_appliance():
#     '''Defines parameters for all appliances of interest'''

#     # Credit: Mingjun Zhong and Jack Kelly
#     return {
#         'kettle':{
#             'windowlength':129,
#             'on_power_threshold':2000,
#             'max_on_power':3998,
#             'mean':700,
#             'std':1000,
#             's2s_length':128
#             },
#         'microwave':{
#             'windowlength':129,
#             'on_power_threshold':200,
#             'max_on_power':3969,
#             'mean':500,
#             'std':800,
#             's2s_length':128},
#         'fridge':{
#             'windowlength':299,
#             'on_power_threshold':50,
#             'max_on_power':3323,
#             'mean':200,
#             'std':400,
#             's2s_length':512},
#         'dishwasher':{
#             'windowlength':599,
#             'on_power_threshold':10,
#             'max_on_power':3964,
#             'mean':700,
#             'std':1000,
#             's2s_length':1536},
#         'washingmachine':{
#             'windowlength':599,
#             'on_power_threshold':20,
#             'max_on_power':3999,
#             'mean':400,
#             'std':700,
#             's2s_length':2000}
#     }

def make_app_params_dict():
    '''Defines parameters for all appliances of interest'''

    # Credit: Jack Kelly
    return {
        'kettle': {
            'max_power': 3100,
            # 'on_power_threshold': 2000,
            'on_power_threshold': 1500,
            'min_on_duration': 12,
            'min_off_duration': 0
            },
        'fridge': {
            'max_power': 300,
            'on_power_threshold': 50,
            'min_on_duration': 60,
            'min_off_duration': 12
            },
        'washing machine': {
            'max_power': 2500,
            'on_power_threshold': 20,
            'min_on_duration': 1800,
            'min_off_duration': 160
            },
        'microwave': {
            'max_power': 3000,
            'on_power_threshold': 200,
            'min_on_duration': 12,
            'min_off_duration': 30
            },
        'dishwasher': {
            'max_power': 2500,
            'on_power_threshold': 10,
            'min_on_duration': 1800,
            'min_off_duration': 1800
            }
    }


def load_house_csv(house_id, dir_refit_csv, nrows=None):
    '''
    Load REFIT CSV for one home.
    '''
    csv_filename = os.path.join(dir_refit_csv, 'CLEAN_House{}.csv'.format(house_id))
    df = pd.read_csv(csv_filename, nrows=nrows)

    return df


def save_refit_data(dir_refit_csv, dir_refit_np, nrows=None, house_ids=None):
    '''
    Load REFIT data from CSVs and saves as numpy arrays.
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
    '''
    Map standardized app name to column to be created in appliances dataframe
    and to the pattern used to identify the appliance in the unstandardized
    name.
    '''
    return OrderedDict([
        ('fridge', {'col': 'Fridge', 'pattern': 'fridge'}),
        ('kettle', {'col': 'Kettle', 'pattern': 'kettle$'}),  # pattern ignores 'Kettle/Toaster'
        ('washing machine', {'col': 'WashingMachine', 'pattern': 'washing *machine'}),
        ('dishwasher', {'col': 'DishWasher', 'pattern': 'dish *washer'}),
        ('microwave', {'col': 'Microwave', 'pattern': '^microwave'})  # pattern ignores 'Combination Microwave'
    ])


def apps_add_cols_from_patterns(apps, app_dict):
    '''
    Add column for each target appliance identifying whether each row
    has an appliance 
    '''
    for (app_name, dct) in app_dict.items():
        app_col = dct['col']
        pattern = dct['pattern']
        apps[app_col] = apps['ApplianceOrig_Raw'].str.lower().str.contains(pattern).astype(int)
    return apps


def create_app_funs(apps, app_dict, app_names):
    '''
    Create functions that require data on appliances. Note that the apps dataframe
    must have the additional appliance columns.
    '''

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

    '''
    Create functions that require the path to the saved REFIT data.
    '''

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
        df['Issues'] = load_issues(house_id)[ts_mask]
    
    return df


def plot_day(house_id, dt, savefile=None, figsize=(9,5), cols=None, title=None):
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
    if title is None:
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
    '''
    Make multiple daily plots for a house within a date range.
    '''
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
    Calculate daily summary stats for each home, to be used in other parts of project,
    e.g., determining which house/date combinations are good/clean enough for the model.
    '''
    
    WATTSEC_2_KWH = 1/3.6e6  # multiply by this to convert from Watt-sec to kWh
    
    # num_unique = lambda x: len(set(x))
    def prop_unchanging_large_value(x):
        '''
        Proportion of elements of x that are unchanging from the last elements and
        above value specified below.
        '''
        if len(x) == 0:
            return None
        else:
            return repeats_above_value(x, 25, True) / len(x)

    # Define functions to be used in aggregation.
    funs = OrderedDict([
        ('RowNum', len),
        ('Time', [min, max]),
        ('UnixDiff', max),
        ('EnergySumOfParts', sum),
        # ('CorrCoef', lambda x: x.max()),
        # ('PctAccountedEnergy', np.std),
        ('Issues', np.mean)
    ])
    for app_num in range(10):
        # Calculate total energy used by appliance.
        # funs['Appliance{}'.format(app_num)] = [np.mean, np.median, min, max, prop_unchanging_large_value]
        funs['Appliance{}'.format(app_num)] = [min, prop_unchanging_large_value]
        funs['EnergyAppliance{}'.format(app_num)] = sum
    
    # Import timestamp data, calculate diffs (for calculating energy used by appliances), and set index (for grouping by day)
    ts_series = load_ts(house_id)
    df = pd.DataFrame({'Unix': ts_series[:nrow]})  # will convert to Time later
    df['UnixDiff'] = df['Unix']  # will take diff later in grouping
    df['Time'] = pd.to_datetime(df['Unix'], unit='s', utc=True)
    del df['Unix']
    df.set_index('Time', inplace=True)
    # df = df.tz_localize('GMT').tz_convert(tz)
    df = df.groupby(pd.TimeGrouper(freq='D')).transform(lambda x: calc_diff(x, True, 6))
    df['Time'] = df.index  # need this twice: one for grouping and one for calculating stats
    df['Issues'] = load_issues(house_id)[:nrow]  # need to do this after diff so they don't get zeroed out (was a bug!)
    df['RowNum'] = range(df.shape[0])  # need to do after diff (though doesn't really matter if taking len())
    
    # Calculate energy used by appliances.
    df['EnergySumOfParts'] = np.zeros(df.shape[0], dtype=np.int)
    for app_num in range(10):
        app_data = load_app(house_id, app_num)[:nrow]
        df['Appliance{}'.format(app_num)] = app_data
        df['EnergyAppliance{}'.format(app_num)] = df['Appliance{}'.format(app_num)] * df['UnixDiff'] * WATTSEC_2_KWH
        if app_num > 0:
            df['EnergySumOfParts'] += df['EnergyAppliance{}'.format(app_num)]
    # df['PctAccountedEnergy'] = df['EnergySumOfParts'] / df['EnergyAppliance0']
    
    # Calculation correlation between sum of appliances and main.
    corr = df[['EnergySumOfParts', 'EnergyAppliance0']].groupby(pd.TimeGrouper(freq='D')).corr().ix[0::2,'EnergyAppliance0']
    corr = corr.reset_index().drop('level_1', axis=1)
    corr.set_index('Time', inplace=True)
    
    # Aggregate by day and apply functions. THIS TAKES THE MOST COMPUTE TIME.
    dstats = df.groupby(pd.TimeGrouper(freq='D')).aggregate(funs)
    
    # Add correlation stats.
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
    '''
    Calculate daily statistics for all homes.
    '''

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
    '''
    Plot day of data for all homes for various days (approx one plot per month).
    '''

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
    Get total energy for the day. When there are multiple cols for an appliance, add energies.
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
    '''
    Add column to indicate whether data should be used.
    '''

    print 'cleaning daily stats...'

    conds = OrderedDict([
        (('RowNum', 'len'), dstats[('RowNum', 'len')] < 500),
        (('UnixDiff', 'max'), dstats[('UnixDiff', 'max')] > 0.25),  # 15 min
        ('HourRange', dstats['HourRange'] < 23.5),
        (('Issues', 'mean'), dstats[('Issues', 'mean')] > 0.1),
        ('HoursInDay', dstats['HoursInDay'] != 24)
        # ('House', dstats['House'].isin([3,11,21]))  # solar panels
    ])
    for app_num in range(10):
        # # Calculate total energy used by appliance.
        # col = ('Appliance{}'.format(app_num), 'mean')
        # conds[col] = dstats[col] < 0

        # col = ('Appliance{}'.format(app_num), 'median')
        # conds[col] = dstats[col] < 0

        col = ('Appliance{}'.format(app_num), 'min')
        conds[col] = dstats[col] < 0
        
        if app_num > 0:
            # Will handle aggreagte later. Want to keep good app signals for synthetic data.
            col = ('Appliance{}'.format(app_num), 'prop_unchanging_large_value')
            conds[col] = dstats[col] > 0.1

        # col = ('EnergyAppliance{}'.format(app_num), 'sum')
        # conds[col] = dstats[col] < 0

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
    '''
    Test whether data for that house and date is okay for use in the model.
    '''
    return dstats.loc[dstats['House']==house_id].loc[str(d)]['Delete'].values[0] == 0


def get_all_dts_for_house(house_id, train_dts=None):
    '''
    Get all dates for the house (since some houses have data that starts earlier or
    later than other houses). Optionally subsets for days to be used in training only,
    which itself only includes days with "good" data; otherwise returns all days, good
    or bad (bad days will be skipped later).
    '''
    
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
    '''
    Return index to align timeseries arrays (one day of data), as well as
    the mask used to subset the entire timeseries (all days) to that of the
    single day.
    '''
    
    dt_end = dt_start + timedelta(days=1)
    ts_mask = get_ts_mask(ts_series=ts_series, dt_start=dt_start)
    ts_day_actual = ts_series[ts_mask]
    ts_day_desired = range(dt2ts(dt_start) + desired_sample_rate,
                           dt2ts(dt_end) + desired_sample_rate,
                           desired_sample_rate)

    return (align_arrays(ts_day_actual, ts_day_desired, padder=0), ts_mask)


def create_real_data(house_ids, app_names, dstats, desired_sample_rate, save_dir=None, is_debug=True):
    '''
    Creates set of real (non-synthetic) data.
    '''
    
    if is_debug:
        print 'creating real data...'

    app_params = make_app_params_dict()

    X = []
    Y1 = []
    Y2 = []
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
            y1 = []
            y2 = []

            for app_name in app_names:

                # Get number of activations for all occurences of target appliance.
                y2_app = 0
                for app_num in app_name2nums[app_name]:
                    app_power = load_app(house_id, app_num)[ts_mask]
                    y2_app += get_num_activations(app_power, ts_series[ts_mask], app_params[app_name])

                y1.append(get_energy(dstats, house_id, dt_start.date(), app_name2nums[app_name]))
                y2.append(y2_app)

            X.append(x)
            x_house.append(house_id)
            x_date.append(dt_start.date())
            Y1.append(y1)
            Y2.append(y2)

        if is_debug:
            t1 = time.time()
            print 'created real data for house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)
    
    if save_dir is not None:
        np.save(os.path.join(save_dir, 'X.npy'), X)
        np.save(os.path.join(save_dir, 'Y1.npy'), Y1)
        np.save(os.path.join(save_dir, 'Y2.npy'), Y2)
        np.save(os.path.join(save_dir, 'x_house.npy'), x_house)
        np.save(os.path.join(save_dir, 'x_date.npy'), x_date)
    
    return X, Y1, Y2, x_house, x_date


def create_bank_choices(dstats, app_names, house_ids, train_dts):

    '''
    Create dataframe of homes, days and appliances to be used when creating
    synthetic data; determine which data is good for swapping with the target
    appliance. E.g., if swap is triggered for a fridge signal, code searches
    this dataframe for a home and date to find an alternative fridge signal.
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
    '''
    Return house, appliance number (i.e., column in REFIT data) and date for
    alternative appliance signal given the bank of choices.
    '''
    
    bc = bank_choices.copy()
    
    if app_name is not None:
        bc = bc.loc[bc['Appliance']==app_name]
    
    row = bc.sample(1)
    dt = dt64_to_datetime(row.index.values[0]).date()
    house_id = row['House'].values[0]
    app_num = row['ApplianceNum'].values[0]
    
    return [house_id, app_num, dt]


def get_aligned_series(house_id, app_num, dt, desired_sample_rate=6):
    '''
    Take house and appliance number (i.e., column in REFIT data) and return
    power series where the timesteps have been standardized to be at the desired
    sample rate.
    '''
    ts_series = load_ts(house_id)
    if isinstance(dt, date):
        dt = date_to_datetime(dt)
    aligned_idx, ts_mask = get_aligned_ts_mask_for_day(ts_series, dt, desired_sample_rate)
    app_series = load_app(house_id, app_num)
    x = app_series[ts_mask][aligned_idx]
    return x, ts_mask, aligned_idx


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
    '''
    Create synthetic data. Only does this for specific houses and datetimes so prevent
    bleeding of information between training and val/test sets. Does this by iterating
    through target appliances and 
    '''

    desired_sample_rate = 6
    bank_choices = create_bank_choices(dstats, app_names, house_ids, train_dts)
    app_params = make_app_params_dict()
    # bank_choices = bank_choices.loc[bank_choices['IsTarget']==1]  # just want target apps

    print 'creating synthetic data...'

    X = []
    Y1 = []
    Y2 = []
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
            y1 = []
            y2 = []

            if is_debug:
                print '        adding target appliance signals...'

            # Iterate over target appliances first so we can create the targets (total daily energy).
            for app_name in app_names:

                app_nums = get_app_nums(house_id, app_name)

                if is_debug:
                    print '        app_name: {}, app_nums: {}'.format(app_name, app_nums)

                if not app_nums:
                    y1.append(0.)
                    y2.append(0.)
                    if is_debug:
                        print '        skipping (adding all zeros for x and y) since there`s no app'
                    continue

                x_app = np.zeros(int(24 * 60 * 60 / desired_sample_rate), dtype=np.int16)
                y1_app = 0.
                y2_app = 0

                for app_num in app_nums:

                    if is_debug:
                        print '        app_name: {}, app_num: {}'.format(app_name, app_num)

                    # If the swap triggers, choose the appliance signal at random
                    # from homes that have that appliance.
                    if np.random.rand() < swap_prob:

                        house_id_rand, app_num_rand, d_rand = get_random_series_metadata(bank_choices, app_name)

                        ts_series_rand = load_ts(house_id_rand)
                        app_power_rand = load_app(house_id_rand, app_num_rand)

                        aligned_idx_rand, ts_mask_rand = get_aligned_ts_mask_for_day(
                            ts_series_rand,
                            date_to_datetime(d_rand),
                            desired_sample_rate)
                        
                        x_app += app_power_rand[ts_mask_rand][aligned_idx_rand]
                        y1_app += get_energy(dstats, house_id_rand, d_rand, [app_num_rand])
                        y2_app += get_num_activations(app_power_rand[ts_mask_rand],
                                                      ts_series_rand[ts_mask_rand],
                                                      app_params[app_name])

                        if is_debug:
                            print '        swapping! random series: house {}, app num {}, date {}'.format(house_id_rand, app_num_rand, d_rand)
                            print '        x_app: {}...'.format(x_app[:5])
                            print '        y1_app: {}'.format(y1_app)
                            print '        y2_app: {}'.format(y2_app)

                    # Otherwise, use actual data from home.
                    else:
                        app_power = load_app(house_id, app_num)

                        x_app += app_power[ts_mask][aligned_idx]
                        y1_app += get_energy(dstats, house_id, dt_start.date(), [app_num])
                        y2_app += get_num_activations(app_power[ts_mask],
                                                      ts_series[ts_mask],
                                                      app_params[app_name])

                        if is_debug:
                            print '        not swapping'
                            print '        x_app: {}...'.format(x_app[:5])
                            print '        y1_app: {}'.format(y1_app)
                            print '        y2_app: {}'.format(y2_app)

                # Add target appliance signal to synthetic aggregate.
                x += x_app
                y1.append(y1_app)
                y2.append(y2_app)

            # Add distractor appliances by iterating through appliances in home.
            if is_debug:
                print '        adding distractor appliance signals...'

            for app_num in range(1, 10):  # don't include main signal

                if is_debug:
                    print '        app_name: {}; app_num: {}'.format(get_app_name(house_id, app_num), app_num)

                # If appliance *is* a target appliance or distractor prob *doesn't*
                # trigger, then skip appliance.
                if is_a_target_app(house_id, app_num) or not np.random.rand() < include_distractor_prob:
                    if is_debug:
                        print '        is either target app or distractor prob doesn`t trigger' 
                    continue
                
                # Othereise, add distractor signal to synthetic aggregate.
                x_app = load_app(house_id, app_num)[ts_mask][aligned_idx]
                x += x_app

                if is_debug:
                    print '        added distractor app'
                    print '        x_app: {}...'.format(x_app[:5])

            if is_debug:
                print '        agggregated x: {}...'.format(x[:5])
                print '        y1`s: {}'.format(y1)
                print '        y2`s: {}'.format(y2)

            X.append(x)
            x_house.append(house_id)
            x_date.append(dt_start.date())
            Y1.append(y1)
            Y2.append(y2)

        # Save data after every home, just in case something crashes.
        # Later, when code is good, it'd be best to do this just once
        # so you don't overwrite the prior files.
        if save_dir is not None:
            np.save(os.path.join(save_dir, 'X.npy'), X)
            np.save(os.path.join(save_dir, 'Y1.npy'), Y1)
            np.save(os.path.join(save_dir, 'Y2.npy'), Y2)
            np.save(os.path.join(save_dir, 'x_house.npy'), x_house)
            np.save(os.path.join(save_dir, 'x_date.npy'), x_date)

    return X, Y1, Y2, x_house, x_date


def get_num_runs(dir_for_model_synth):
    '''
    Gets number of times the synthetic data was run.
    '''
    num_runs = 0
    for filename in os.listdir(dir_for_model_synth):
        try:
            filename_int = int(filename)
            num_runs = max(num_runs, filename_int)
        except ValueError:
            pass
    return num_runs


def get_train_dts(dstats, prop_train, save_dir=None, is_debug=False):
    '''
    Randomly samples datetimes to be used in training. Saves it so it can be referenced later.
    '''

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


def get_activations(chunk, min_off_duration=0, min_on_duration=0,
                    border=1, on_power_threshold=5, max_power=None):
    """
    Credit: Jack Kelly and NILMTK package.
    
    Included max_power=None just so that params could be entered as
    kwargs.
    
    https://github.com/nilmtk/nilmtk/blob/master/nilmtk/electric.py
    
    
    Returns runs of an appliance.
    Most appliances spend a lot of their time off.  This function finds
    periods when the appliance is on.
    Parameters
    ----------
    chunk : pd.Series
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.
    border : int
        Number of rows to include before and after the detected activation
    on_power_threshold : int or float
        Watts
    Returns
    -------
    list of pd.Series.  Each series contains one activation.
    """
    when_on = chunk >= on_power_threshold

    # Find state changes
    state_changes = when_on.astype(np.int8).diff()
    del when_on
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    del state_changes

    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []

    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    assert len(switch_on_events) == len(switch_off_events)

    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = (chunk.index[switch_on_events[1:]].values -
                         chunk.index[switch_off_events[:-1]].values)

        off_durations = timedelta64_to_secs(off_durations)

        above_threshold_off_durations = np.where(
            off_durations >= min_off_duration)[0]

        # Now remove off_events and on_events
        switch_off_events = switch_off_events[
            np.concatenate([above_threshold_off_durations,
                            [len(switch_off_events)-1]])]
        switch_on_events = switch_on_events[
            np.concatenate([[0], above_threshold_off_durations+1])]
    assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = (chunk.index[off] - chunk.index[on]).total_seconds()
        if duration < min_on_duration:
            continue
        on -= 1 + border
        if on < 0:
            on = 0
        off += border
        activation = chunk.iloc[on:off]
        # throw away any activation with any NaN values
        if not activation.isnull().values.any():
            activations.append(activation)

    return activations


def get_num_activations(app_power, ts_series, app_params):
    
    # Chunk is NILM terminology for a power series.
    chunk = pd.Series(
        app_power,
        index=pd.to_datetime(ts_series, unit='s', utc=True)
    )
    
    # Get list of series that have activations.
    activations = get_activations(chunk, border=1, **app_params)
    
    return len(activations)




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
    dir_for_model = os.path.join(dir_data, 'for_model')

    dir_refit_csv = os.path.join(dir_data, 'CLEAN_REFIT_081116')
    dir_refit = os.path.join(dir_data, 'refit')
    
    path_apps = os.path.join(dir_data, 'appliances.csv')
    path_daily_stats = os.path.join(dir_data, 'stats_by_day.pkl')

    HOUSE_IDS = range(1, 22); HOUSE_IDS.remove(14)  # no house 14
    APP_NAMES = ['fridge', 'kettle', 'washing machine', 'dishwasher', 'microwave']
    HOUSE_IDS_TEST = [2,9,20]
    HOUSE_IDS_TRAIN_VAL = [house_id for house_id in HOUSE_IDS if house_id not in HOUSE_IDS_TEST]
    HOUSE_IDS_SOLAR = [3,11,21]
    HOUSE_IDS_NOT_SOLAR = [house_id for house_id in HOUSE_IDS if house_id not in HOUSE_IDS_SOLAR]
    # TRAIN_VAL_DATE_MAX = datetime(2015,2,28)

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
    # train_dts = get_train_dts(dstats, prop_train, save_dir=None)
    # train_dts = np.load(os.path.join(dir_run, 'train_dts.npy'))

    # X, Y1, Y2, x_house, x_date = create_real_data(HOUSE_IDS, APP_NAMES, dstats, desired_sample_rate, dir_run_real)  # can remove solar later

    # Create synthetic data.
    for run_num in range(1,synthetic_data_runs+1):
        print '=============== RUN NUM {} ==============='.format(run_num)
        X, Y1, Y2, x_house, x_date = create_synthetic_data(
            dstats,
            HOUSE_IDS_TRAIN_VAL,
            train_dts,
            APP_NAMES,
            swap_prob,
            include_distractor_prob,
            save_dir=os.path.join(dir_run_synthetic, str(run_num)),
            is_debug=False
            )

    # X, Y1, Y2, x_house, x_date = create_synthetic_data(
    #     dstats,
    #     HOUSE_IDS_TRAIN_VAL,
    #     train_dts[-5:],
    #     APP_NAMES,
    #     swap_prob,
    #     include_distractor_prob,
    #     save_dir=os.path.join(dir_run_synthetic, str(1)),
    #     is_debug=True
    #     )

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