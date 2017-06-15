
import os
import shutil
import numpy as np
import pandas as pd
import time
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
        # appliance_cols = app.loc[app['House'] == house_id, 'Appliance'].values.tolist()
        # columns = ['Timestamp', 'aggregate'] + appliance_cols
        # columns = ['timestamp'] + ['appliance{}'.format(i) for i in range(10)]
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


def apps_add_cols_from_patterns(app_dict):
    for (app_name, dct) in app_dict.items():
        app_col = dct['col']
        pattern = dct['pattern']
        apps[app_col] = apps['ApplianceOrig'].str.lower().str.contains(pattern).astype(int)
    return apps


def create_app_funs(apps, app_dict):

    def get_house_app_tuples(app_name, return_pd=False):
        app_col = app_dict[app_name]['col']
        if return_pd:
            return apps.loc[apps[app_col] == 1]
        house_to_app = apps.loc[apps[app_col] == 1][['House', 'ApplianceNum']].values
        return [(h, a) for h, a in house_to_app]

    def get_app_nums(house_id, app_name):
        app_col = app_dict[app_name]['col']==1
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
    
    # if include_time:
    #     df['Time'] = pd.to_datetime(df['Unix'], unit='s', utc=True)
    
    return df


def plot_day(house_id, dt, savefile=None, figsize=(7,5)):
    '''
    Plot time series of power data for each appliance, for specified house and date(time).
    '''
    df = get_df(house_id, use_app_names=True, dt_start=dt)
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


def calc_stats_for_house(house_id, nrow=None):
    '''
    Calculate daily summary stats for each home and some appliance in some home.
    '''
    
    # Define functions to be used in aggregation.
    funs = OrderedDict([
        ('RowNum', len),
        ('Time', [min, max]),
        ('UnixDiff', max),
        ('EnergySumOfParts', sum),
        # ('CorrCoef', lambda x: x.max()),
        ('PctAccountedEnergy', np.std)
    ])
    for app_num in range(10):
        # Calculate total energy used by appliance.
        funs['EnergyAppliance{}'.format(app_num)] = sum
    
    # Import timestamp data, calculate diffs (for calculating energy used by appliances), and set index (for grouping by day)
    ts_series = load_ts(house_id)
    df = pd.DataFrame({'Unix': ts_series})  # will convert to Time later
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
        df['EnergyAppliance{}'.format(app_num)] = df['Appliance{}'.format(app_num)] * df['UnixDiff']
        if app_num > 0:
            df['EnergySumOfParts'] += df['EnergyAppliance{}'.format(app_num)]
    df['PctAccountedEnergy'] = df['EnergySumOfParts'] / df['EnergyAppliance0']
    
    # Calculation correlation between sum of appliances and main.
    corr = df[['EnergySumOfParts', 'EnergyAppliance0']].groupby(pd.TimeGrouper(freq='D')).corr().ix[0::2,'EnergyAppliance0']
    corr = corr.reset_index().drop('level_1', axis=1)
    corr.set_index('Time', inplace=True)
    
    # Aggregate by day and apply functions.
    stats = df.groupby(pd.TimeGrouper(freq='D')).aggregate(funs)
    assert corr.shape[0] == stats.shape[0]
    stats['SumToMainCorr'] = corr['EnergyAppliance0']
    
    # Calculate percent of aggregate energy accounted for by appliances.
    stats[('PctAccountedEnergy', 'total')] = stats[('EnergySumOfParts', 'sum')] / stats[('EnergyAppliance0', 'sum')]
    
    # Get timestamp range in hours.
    stats['HourRange'] = stats[('Time', 'max')] - stats[('Time', 'min')]
    stats['HourRange'] = stats['HourRange'] / np.timedelta64(1, 'h')
    
    # Make TimestampDiff in hours.
    stats[('UnixDiff', 'max')] = stats[('UnixDiff', 'max')]/60/60
    
    return stats


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


def create_daily_stats(house_ids, pkl_path=None, nrow=None):

    stats = []
    for house_id in house_ids:
        print 'calculating stats for house {}...'.format(house_id)
        stats_house = calc_stats_for_house(house_id, nrow=nrow)
        stats_house['House'] = house_id
        stats.append(stats_house)
    stats = pd.concat(stats)
    print 'done!'
    
    if pkl_path is not None:
        stats.to_pickle(pkl_path)
    
    return stats


if __name__ == '__main__':

    pass