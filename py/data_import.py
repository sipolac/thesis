
import os
import numpy as np
import pandas as pd
import time


def calc_diff(df_col, pad_end=True, nan_fill=None):
    '''
    Calculate diff of array of ints, padding beginning or end with nan_fill
    '''
    beg = df_col.diff()[1:]
    end = nan_fill
    if not pad_end:
        beg, end = end, beg  # pad beginning instead
    return np.append(beg, end).astype(int)


def load_house_csv(house_id, refit_data_path, columns, nrows=None):
    '''
    Inspired by code in NILMTK package.
    '''
    csv_filename = os.path.join(refit_data_path, 'House{}.csv'.format(house_id))
    df = pd.read_csv(csv_filename, names=columns, nrows=nrows)
    df['House'] = house_id
    df = df[['Timestamp', 'House'] + ['Appliance{}'.format(i) for i in range(10)]]  # reorder cols

    return df


def save_refit_data_to_sqlite(conn, refit_data_path, app, nrows=None, chunksize=10000, is_debug=True):
    '''
    '''

    house_ids = range(1, 22)
    house_ids.remove(14)  # no house 14


    if is_debug:
        print 'writing REFIT data to SQLite database...'

    for house_id in house_ids:
        t0 = time.time()
        # appliance_cols = app.loc[app['House'] == house_id, 'Appliance'].values.tolist()
        # columns = ['Timestamp', 'aggregate'] + appliance_cols
        columns = ['Timestamp'] + ['Appliance{}'.format(i) for i in range(10)]
        df = load_house_csv(house_id, refit_data_path, columns, nrows)
        if_exists = 'replace' if house_id == 1 else 'append'
        df.to_sql('power', conn, if_exists=if_exists, index=False, chunksize=chunksize)
        t1 = time.time()
        if is_debug:
            print 'added house {0} to the database ({1:.2g} min)'.format(house_id, (t1 - t0)/60)

    print 'done!'
    
    conn.close()
