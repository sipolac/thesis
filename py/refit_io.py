
import os
import shutil
import numpy as np
import pandas as pd
import time


def load_house_csv(house_id, refit_data_path, columns, nrows=None):
    '''
    '''
    csv_filename = os.path.join(refit_data_path, 'house{}.csv'.format(house_id))
    df = pd.read_csv(csv_filename, names=columns, nrows=nrows)
    # df['house'] = house_id
    # df = df[['timestamp', 'house'] + ['appliance{}'.format(i) for i in range(10)]]  # reorder cols

    return df


def save_refit_data(refit_raw_path, method='numpy', dir_refit=None, dir_data=None, nrows=None, conn=None, chunksize=10000):
    '''
    '''

    method = method.lower()
    assert method in ['numpy', 'sqlite']

    house_ids = range(1, 22)
    house_ids.remove(14)  # no house 14

    print 'writing REFIT data...'

    for house_id in house_ids:

        t0 = time.time()

        # Load data for a house.
        # appliance_cols = app.loc[app['House'] == house_id, 'Appliance'].values.tolist()
        # columns = ['Timestamp', 'aggregate'] + appliance_cols
        columns = ['timestamp'] + ['appliance{}'.format(i) for i in range(10)]
        df = load_house_csv(house_id, refit_raw_path, columns, nrows)
        
        if method == 'sqlite':
            if_exists = 'replace' if house_id == 1 else 'append'
            df.to_sql('power', conn, if_exists=if_exists, index=False, chunksize=chunksize)
        
        elif method == 'numpy':
            if house_id == 1:
                # Create directory where we'll save REFIT data.
                if os.path.exists(dir_refit):
                    shutil.rmtree(dir_refit)
                os.makedirs(dir_refit)

            # Make directory for house.
            dir_house = os.path.join(dir_refit, 'house{}'.format(house_id))
            os.makedirs(dir_house)
            for column in columns:
                # Data type should be int32 for timestamp and int16 for appliance power levels.
                dtype = np.int32 if column == 'timestamp' else np.int16
                np.save(os.path.join(dir_house, '{}.npy'.format(column)), df[column].values.astype(dtype))

        t1 = time.time()
        print 'added house {0} ({1:.2g} min)'.format(house_id, (t1 - t0)/60)

    print 'done!'
    
    conn.close()



# def query(string, dir_data):
#     conn = sqlite3.connect(os.path.join(dir_data, 'refit.db'))
#     df = pd.read_sql(string, conn)
#     conn.close()
#     return df
