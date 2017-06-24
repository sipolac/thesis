
from __future__ import division

from utils import *
from create_data import *

import os
import pandas as pd
import numpy as np

from datetime import datetime
from datetime import timedelta
from datetime import date

import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout
from keras import backend as K

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

matplotlib.style.use('ggplot')


def load_real_data(dir_for_model_real):
    '''
    Load data where aggregate signal is not synthesized. Include all homes;
    subsetting (e.g., excluding homes w/ solar panels) can be done later.
    '''

    X = np.load(os.path.join(dir_for_model_real, 'X.npy'))
    Y = np.load(os.path.join(dir_for_model_real, 'Y.npy'))
    x_house = np.load(os.path.join(dir_for_model_real, 'x_house.npy'))
    x_date = np.load(os.path.join(dir_for_model_real, 'x_date.npy'))

    return X, Y, x_house, x_date


def load_synth_data(dir_for_model_synth, save=False):
    '''
    Load synthetic data. Must take from multiple directories, one for each run.
    '''

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


def show_data_dims(all_data):
    '''
    Debug function for printing out dimensions of 
    '''
    for key in all_data.keys():
        print key, type(all_data[key])
        for dat in all_data[key]:
            print dat.shape


def remove_solar_from_real(all_data, house_ids_solar, x_house_idx):
    '''
    Remove data where house has solar panels.
    '''
    assert 'real' in all_data.keys()
    solar_mask = np.in1d(all_data['real'][x_house_idx], house_ids_solar)
    for i in range(len(all_data['real'])):
        all_data['real'][i] = all_data['real'][i][~solar_mask]
    return all_data


def split_real_into_train_and_valtest(all_data, house_ids_train_val, train_dates, x_house_idx, x_date_idx):
    '''
    Split real data into training and val/test. Deletes the "real" dataset afterward.
    
    Note that the "_idx" values must be defined either at time of function creation
    or in the global environment.
    '''
    assert 'real' in all_data.keys()
    
    # Split real data into train and val/test (same dataset for now).
    train_mask_real = (np.in1d(all_data['real'][x_date_idx], train_dates)) & \
                      (np.in1d(all_data['real'][x_house_idx], house_ids_train_val))
    n_train_real = all_data['real'][0][train_mask_real].shape[0]  # can choose any data set in "real"
    print 'obs for train/val: {} ({:0.2g}% of total)'.format(
        n_train_real,
        n_train_real / all_data['real'][0].shape[0] * 100
    )
    all_data['real_train'] = []
    all_data['val_test'] = []  # test doesn't take synthetic data, so don't need "real_" suffix
    for dat in all_data['real']:
        all_data['real_train'].append(dat[train_mask_real])
        all_data['val_test'].append(dat[~train_mask_real])
    del all_data['real']
    
    return all_data


def split_valtest_into_val_and_test(all_data, x_house_idx, val_test_size=0.5):
    '''
    Split val/test data into validation and test data. This is all real (non-synthetic).
    '''
    assert 'val_test' in all_data.keys()
    val_test_split = train_test_split(*all_data['val_test'],
                                       train_size=val_test_size,
                                       stratify=all_data['val_test'][x_house_idx])
    all_data['val'] = val_test_split[::2]
    all_data['test'] = val_test_split[1::2]
    del all_data['val_test']
    return all_data

def create_scalers_for_real_synth_both(all_data, X_idx):
    '''
    Create scaler/standardizer for real data and synthetic data, and then
    both when they're combined.
    '''
    # Create scalers for real and synthetic (mean only since want to preserve differences between
    # data points).
    scaler_real = StandardScaler(with_std=False).fit(all_data['real_train'][X_idx].reshape(-1,1))
    scaler_synth = StandardScaler(with_std=False).fit(all_data['synth_train_all'][X_idx].reshape(-1,1))

    # Create scaler for combined real and synthetic (standard dev only
    # since data was already demeaned separately). Sample synthetic
    # so that there are the same number of obs for real and synthetic.
    n_train_real = all_data['real_train'][0].shape[0] 
    sample_idx = np.random.choice(all_data['synth_train_all'][0].shape[0], n_train_real, replace=False)
    train_dat = np.concatenate((all_data['synth_train_all'][X_idx][sample_idx],
                                all_data['real_train'][X_idx]))
    scaler_both = StandardScaler(with_mean=False).fit(train_dat.reshape(-1,1))
    del train_dat
    
    return scaler_real, scaler_synth, scaler_both


def create_training_data(all_data, scaler_real, scaler_synth, scaler_both, X_idx, x_house_idx, synthetic_only=False, random_state=None):
    '''
    Combine real and sampled synthetic training data.
    '''
    n_train_real = all_data['real_train'][0].shape[0] 
    if synthetic_only:
        n_train_real *= 2

    # Randomly grab some of the synthetic data for training,
    # making sure it's the same size as the real data. Ignore
    # the rest of the split.
    all_data['synth_train'] = train_test_split(
        *all_data['synth_train_all'],
        train_size=n_train_real,
        stratify=all_data['synth_train_all'][x_house_idx],  # by house
        random_state=random_state)[::2]  # takes the first of the output pairs, which has same # obs as real train

    all_data['train'] = []
    for i, (r, s) in enumerate(zip(all_data['real_train'], all_data['synth_train'])):

        if i == X_idx:
            # Scale mean of real and synthetic series separately since levels of real is higher.
            r = scaler_real.transform(r)
            s = scaler_synth.transform(s)

        # Sort of for debugging only: want to see if synth does better than real + synth.
        if synthetic_only:
            all_data['train'].append(s)
            continue

        all_data['train'].append(np.concatenate((r, s)))

    del all_data['synth_train']

    # Now scale combined real/synthetic power series by std.
    all_data['train'][X_idx] = scaler_both.transform(all_data['train'][X_idx])    

    # Mix up real and synth obs.
    all_data['train'] = shuffle(*all_data['train'], random_state=random_state)

    return all_data['train']


def take_diff_df(X):
    '''
    Diff each row of X.
    '''
    X = np.diff(X)
    zs = np.zeros((X.shape[0], 1), dtype=int)
    X = np.concatenate((zs, X), axis=1)
    return X


def get_extreme_mask(X, q):
    '''
    Outputs mask that returns obs w/ extreme values for any of the features of array X.
    Note the broadcasting in the function: each feature is treated separately.
    '''
    extreme_mask = np.any(X > np.percentile(X, q=q, axis=0), axis=1)
    return extreme_mask

def remove_extremes(data_set, q, Y_idx):
    '''
    Takes data_set (list of arrays X, Y, x_house, ...) and removes obs where values in
    data as identified by Y_idx are extreme (according to cutoff q).
    '''
    extreme_mask = get_extreme_mask(data_set[Y_idx], q=q)
    new_data = []
    for dat in data_set:
        new_data.append(dat[~extreme_mask])
    return new_data


def scale_rows_of_array(X, with_mean, with_std):
    '''
    Scales the rows of a 2D array individually.
    '''
    return StandardScaler(with_mean=with_mean, with_std=with_std).fit_transform(X.T).T


def reshape_as_tensor(list_of_Xs):
    '''
    Reshape input 2D dataframe (shape = (obs, features)) as tensor.
    '''
    image_data_format = K.image_data_format()
    assert image_data_format in ['channels_first', 'channels_last']
    list_of_tensors = []
    for X in list_of_Xs:
        if image_data_format == 'channels_last':  # default on dev machine
            X = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        list_of_tensors.append(X)
    return list_of_tensors


def get_chunk(X, chunk_num, total_chunks):
    '''
    Get chunks of array X by row. Chunks can be different sizes.
    '''
    assert chunk_num < total_chunks  # b/c of zero-indexing
    n_per_chunk = X.shape[0] / total_chunks
    start_idx = int(chunk_num * n_per_chunk)
    end_idx = int((chunk_num * n_per_chunk) + n_per_chunk)
    return X[start_idx:end_idx]


def extract_targets(all_data, split_type, app_name, app_names, Y_idx):
    '''
    Simply pick the right column of targets.
    '''
    assert split_type in ['train', 'val']
    app_idx = APP_NAMES.index(app_name)
    Y = all_data[split_type][Y_idx]
    y = [Y_row[app_idx] for Y_row in Y]
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    return y


def plot_empir_cum(x):
    '''Plot empirical cumulative distribution'''
    # https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    return plt.step(sorted(x), np.arange(len(x))/len(x), color='black')


if __name__ == '__main__':

    N_PER_DAY = 14400  # 24 * 60 * 60 / 6
    HOUSE_IDS = range(1, 22); HOUSE_IDS.remove(14)  # no house 14
    HOUSE_IDS_TEST = [2,9,20]
    HOUSE_IDS_TRAIN_VAL = [house_id for house_id in HOUSE_IDS if house_id not in HOUSE_IDS_TEST]
    # HOUSE_IDS_SOLAR = [3,11,21]  # according to paper
    HOUSE_IDS_SOLAR = [1,11,21]  # according to inspection
    HOUSE_IDS_NOT_SOLAR = [house_id for house_id in HOUSE_IDS if house_id not in HOUSE_IDS_SOLAR]
    # TRAIN_VAL_DATE_MAX = date(2015,2,28)
    APP_NAMES = ['fridge', 'kettle', 'washing machine', 'dishwasher', 'microwave']
    TRAIN_DTS = np.load(os.path.join(dir_for_model, 'train_dts.npy'))

