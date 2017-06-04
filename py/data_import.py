
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
    	beg, end = end, beg
    return np.append(beg, end).astype(int)


def load_house_csv(house_id, refit_data_path, app, nrows=None):
    '''
    Inspired by code in NILMTK package.
    '''
    
    tz = 'Europe/London'  # London timezone since REFIT has UK houses
    
    # Load data, making sure that columns have correct appliance names for house.
    csv_filename = os.path.join(refit_data_path, 'House{}.csv'.format(house_id))
    appliance_cols = app.loc[app['House'] == house_id, 'Appliance'].values.tolist()
    columns = ['Timestamp', 'aggregate'] + appliance_cols
    df = pd.read_csv(csv_filename, names=columns, nrows=nrows)
    
    # Melt data so appliances and power values are long instead of wide.
    df = pd.melt(df, id_vars=['Timestamp'], var_name='Appliance', value_name='Power')
    
    # Add differenced values for calculating total energy (energy = power * time).
    df['PowerDiff'] = df['Power'].groupby(df['Appliance']).transform(lambda x: calc_diff(x, False, 0.))
    df['Duration'] = df['Timestamp'].groupby(df['Appliance']).transform(lambda x: calc_diff(x, True, 6.))

    # Convert the integer index column to timezone-aware datetime.
    df['Timestamp'] = pd.to_datetime(df.Timestamp, unit='s', utc=True)
    df.set_index('Timestamp', inplace=True)
    df = df.tz_localize('GMT').tz_convert(tz)
    # Add date info for easier subsetting later.
    # df['Year'] = pd.DatetimeIndex(df.index).year
    # df['Month'] = pd.DatetimeIndex(df.index).month
    # df['Day'] = pd.DatetimeIndex(df.index).day
    # df['Hour'] = pd.DatetimeIndex(df.index).hour
    df.index = np.array(df.index, dtype=np.datetime64)
    df.index.name = 'Timestamp'  # overwrites name 'index'
    df.reset_index(inplace=True)

    df['House'] = house_id

    df = df[['Timestamp', 'House', 'Appliance', 'Power', 'PowerDiff', 'Duration']]  # reorder cols

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
		df = load_house_csv(house_id, refit_data_path, app, nrows=nrows)
		if_exists = 'replace' if house_id == 1 else 'append'
		df.to_sql('power', conn, if_exists=if_exists, index=False, chunksize=chunksize)
		t1 = time.time()
		if is_debug:
			print 'added house {0} to the database ({1:.2g} min)'.format(house_id, (t1 - t0)/60)
        
	print 'done!'
    
	conn.close()
