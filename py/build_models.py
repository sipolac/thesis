
from __future__ import division

from utils import *
from create_data import *

import os
import pandas as pd
import numpy as np
import re

from datetime import datetime, date, timedelta

import cPickle as pickle

import matplotlib.pyplot as plt
import matplotlib

import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
    Y1 = np.load(os.path.join(dir_for_model_real, 'Y1.npy'))
    Y2 = np.load(os.path.join(dir_for_model_real, 'Y2.npy'))
    x_house = np.load(os.path.join(dir_for_model_real, 'x_house.npy'))
    x_date = np.load(os.path.join(dir_for_model_real, 'x_date.npy'))

    return X, Y1, Y2, x_house, x_date


def load_synth_data(dir_for_model_synth, save=False):
    '''
    Load synthetic data. Must take from multiple directories, one for each run.
    '''

    num_runs = get_num_runs(dir_for_model_synth)

    X = []
    Y1 = []
    Y2 = []
    x_house = []
    x_date = []
    for run_num in range(1,num_runs+1):

        dir_run_num = os.path.join(dir_for_model_synth, '{}'.format(run_num))

        X_run = np.load(os.path.join(dir_run_num, 'X.npy'))
        Y1_run = np.load(os.path.join(dir_run_num, 'Y1.npy'))
        Y2_run = np.load(os.path.join(dir_run_num, 'Y2.npy'))
        x_house_run = np.load(os.path.join(dir_run_num, 'x_house.npy'))
        x_date_run = np.load(os.path.join(dir_run_num, 'x_date.npy'))

        X.append(X_run)
        Y1.append(Y1_run)
        Y2.append(Y2_run)
        x_house.append(x_house_run)
        x_date.append(x_date_run)

    X = np.concatenate(X)  # cuts down to correct number of obsw
    Y1 = np.concatenate(Y1)
    Y2 = np.concatenate(Y2)
    x_house = np.concatenate(x_house)
    x_date = np.concatenate(x_date)

    if save:
        np.save(os.path.join(dir_for_model_synth, 'X.npy'), X)
        np.save(os.path.join(dir_for_model_synth, 'Y1.npy'), Y1)
        np.save(os.path.join(dir_for_model_synth, 'Y2.npy'), Y2)
        np.save(os.path.join(dir_for_model_synth, 'x_house.npy'), x_house)
        np.save(os.path.join(dir_for_model_synth, 'x_date.npy'), x_date)
    
    return X, Y1, Y2, x_house, x_date


def get_bad_data_tups(dstats, dstats_cond):
    
    dstats['Delete_BadAgg'] = 0
    dstats.loc[dstats_cond, 'Delete_BadAgg'] = 1
    bad_obs_df = dstats.loc[dstats['Delete_BadAgg'] == 1, ['House']]
    bad_obs_df.reset_index(inplace=True)

    dts = bad_obs_df['Time']
    house_ids = bad_obs_df['House']

    tups = []
    for house_id, dt in zip(house_ids, dts):
        tups.append((house_id, dt.date()))

    return tups


def remove_tups(data_set, tups, x_house_idx, x_date_idx, is_debug=True):
    '''
    data_set is a list of arrays X, Y1, etc. and tupes is a list of 
    tuples with elements (house_id, datetime). Removes observations
    from arrays where house_id and date(time)s are in the tuple.
    '''
    
    # Find indices 
    bad_idx = []
    for house_id, dt in tups:
        bad_idx.append(np.where(((data_set[x_house_idx] == house_id) & (data_set[x_date_idx] == dt)))[0])
    bad_idx = np.concatenate(bad_idx)
    good_idx = [x for x in range(data_set[0].shape[0]) if x not in bad_idx]

    # Subset each part of data set in list and return new list.
    dat_new = []
    for dat in data_set:
        dat_new.append(dat[good_idx])
    
    if is_debug:
        obs_before = data_set[0].shape[0]
        obs_after = dat_new[0].shape[0]
        obs_prop_diff = (obs_before - obs_after) / dat_new[0].shape[0]
        print 'removed {0} obs ({1:0.2g}% of total)'.format(obs_before - obs_after, obs_prop_diff*100)

    return dat_new


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
    '''
    assert 'real' in all_data.keys()
    
    # Split real data into train and val/test (same dataset for now).
    train_mask_real = (np.in1d(all_data['real'][x_date_idx], train_dates)) & \
                      (np.in1d(all_data['real'][x_house_idx], house_ids_train_val))
    n_train_real = sum(train_mask_real)
    print 'real obs for training: {} ({:0.2g}% of total)'.format(
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

def prepare_real_data(dir_for_model_real,
                      dstats,
                      extreme_percentile_cutoff,
                      house_ids_solar,
                      house_ids_train_val,
                      train_dates,
                      X_idx, Y1_idx, Y2_idx, x_house_idx, x_date_idx,
                      all_data = OrderedDict()):

    '''
    Process real data for model.
    '''

    print 'processing real data...'
    all_data['real'] = list(load_real_data(dir_for_model_real))  # as list so can alter elements

    print 'removing extreme values...'
    for idx in [Y1_idx, Y2_idx]:
        all_data['real'] = remove_extremes(all_data['real'], extreme_percentile_cutoff, idx)

    print 'removing homes with solar panels...'
    all_data = remove_solar_from_real(all_data, house_ids_solar, x_house_idx)

    # Do this after solar to see if it really makes a difference since corr is low w/ solar houses.
    print 'remove obs where correlation between main and sum of apps is low'
    corr_tups = get_bad_data_tups(dstats, dstats['SumToMainCorr'] < 0.1)
    all_data['real'] = remove_tups(all_data['real'], corr_tups, x_house_idx, x_date_idx)

    print 'remove obs where agg value is repeated'
    repeat_cond = dstats[('Appliance0', 'prop_unchanging_large_value')] > 0.1
    corr_tups = get_bad_data_tups(dstats, repeat_cond)
    all_data['real'] = remove_tups(all_data['real'], corr_tups, x_house_idx, x_date_idx)

    print 'splitting into training, validation and test data...'
    all_data = split_real_into_train_and_valtest(all_data, house_ids_train_val, train_dates,
                                                 x_house_idx, x_date_idx)
    all_data = split_valtest_into_val_and_test(all_data, x_house_idx)
    
    print 'datasets: {}'.format(all_data.keys())
    
    return all_data


def prepare_synth_data(dir_for_model_synth,
                       extreme_percentile_cutoff,
                       X_idx, Y1_idx, Y2_idx, x_house_idx, x_date_idx,
                       all_data = OrderedDict()):

    print 'loading synthetic data...'
    all_data['synth_train_all'] = list(load_synth_data(dir_for_model_synth))

    print 'removing extreme values...'
    for idx in [Y1_idx, Y2_idx]:
        all_data['synth_train_all'] = remove_extremes(all_data['synth_train_all'], extreme_percentile_cutoff, idx)

    print 'datasets: {}'.format(all_data.keys())
        
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
    X_train = np.concatenate((all_data['synth_train_all'][X_idx][sample_idx],
                             all_data['real_train'][X_idx]))
    scaler_both = StandardScaler(with_mean=False).fit(X_train.reshape(-1,1))
    del X_train
    
    return scaler_real, scaler_synth, scaler_both


def take_row_diffs(X):
    '''
    Diff each row of X and pad with zeros.
    '''
    X = np.diff(X)
    zs = np.zeros((X.shape[0], 1), dtype=int)
    X = np.concatenate((zs, X), axis=1)
    return X


# model_name = 'pilot_model_tmp'

def create_model(
    num_outputs,
    output_layer_activation,
    n_per_day,
    num_conv_layers,
    num_dense_layers,
    start_filters,
    deepen_filters,
    kernel_size,
    strides,
    dilation_rate,
    do_pool,
    pool_size,
    last_dense_layer_size,
    dropout_rate_after_conv,
    dropout_rate_after_dense,
    use_batch_norm,
    optimizer,
    learning_rate,
    l2_penalty
):
    
    if dilation_rate > 1:
        # Make dilation rate override stride length.
        strides = 1
    
    if strides > kernel_size:
        strides = kernel_size
    
    if not do_pool:
        pool_size = None
        
    if num_dense_layers == 0:
        last_dense_layer_size == None
        
    assert not (dilation_rate != 1 and strides != 1)

    kernel_regularizer = regularizers.l2(l2_penalty)
    input_shape = (N_PER_DAY, 1) if K.image_data_format() == 'channels_last' else (1, N_PER_DAY)

    model = Sequential()
    
    # Add convolutional layers.
    for layer_num in range(num_conv_layers):
        
        filter_multiplier = 2**layer_num if deepen_filters else 1
        filters = start_filters * filter_multiplier
        
        conv_args = {'filters': filters,
                     'kernel_size': kernel_size,
                     'strides': strides,
                     'padding': 'same',
                     'dilation_rate': dilation_rate,
                     'activation': 'relu',
                     'name': 'conv_{}'.format(layer_num),
                     'kernel_regularizer': kernel_regularizer}
        if layer_num == 0:
            # Need input shape if first layer.
            conv_args['input_shape'] = input_shape
        
        model.add(Conv1D(**conv_args))
        
        if do_pool:
            model.add(MaxPooling1D(pool_size, padding='same', name='pool_{}'.format(layer_num)))
    
    model.add(Dropout(dropout_rate_after_conv, name='dropout_after_conv'))
    model.add(Flatten(name='flatten'))
    
    # Add dense layers.
    for layer_num in range(num_dense_layers):
        layer_size = last_dense_layer_size * 2**(num_dense_layers - layer_num - 1)
        model.add(Dense(layer_size, activation='relu',
                        kernel_regularizer=kernel_regularizer, name='dense_{}'.format(layer_num)))
        model.add(Dropout(dropout_rate_after_dense, name='dropout_dense_{}'.format(layer_num)))
        if use_batch_norm:
            model.add(BatchNormalization(name='batch_norm_{}'.format(layer_num)))
    
    model.add(Dense(num_outputs, activation=output_layer_activation, name='dense_output',
                    kernel_regularizer=kernel_regularizer))
    
    model.compile(loss='mean_squared_error',
                  # optimizer='adam',
                  optimizer = optimizer(lr=learning_rate),
                  metrics=None)
    
    return model


def get_scale_vars_Y(Y):
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    return Y_mean, Y_std


def scale_Y(Y, Y_mean=None, Y_std=None):
    if Y_mean is None or Y_std is None:
        Y_mean, Y_std = get_scale_vars_Y(Y)
    Y_scaled = (Y - Y_mean) / Y_std
    return Y_scaled


def unscale_Y(Y_scaled, Y_mean, Y_std):
    return Y_scaled * Y_std + Y_mean


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


def reshape_as_tensor(X):
    '''
    Reshape input 2D dataframe (shape = (obs, features)) as tensor.
    '''
    image_data_format = K.image_data_format()
    assert image_data_format in ['channels_first', 'channels_last']
    if image_data_format == 'channels_last':  # default on dev machine
        # X = X.reshape(X.shape[0], X.shape[1], 1)
        X = np.expand_dims(X, 2)
    else:
        # X = X.reshape(X.shape[0], 1, X.shape[1])
        X = np.expand_dims(X, 1)
    return X


def generate_data(
    real_tup, synth_tup,
    scaler_real=None, scaler_synth=None, scaler_both=None, target_scaler=None,
    do_shuffle=True, random_state=None,  # can turn off shuffling for debugging
    batch_size=32,
    real_to_synth_ratio=0.5,
    as_tensor=True,
    use_saved=False,
    save_dir=None
):
    '''
    Combine real and sampled synthetic training data.
    '''

    # real_path = os.path.join(save_dir, 'real_generator.pkl')
    # synth_path = os.path.join(save_dir, 'synth_generator.pkl')

    # if use_saved:
    #     print 'loading saved data for generator...'
    #     real_tup = pickle.load(open(real_path))
    #     synth_tup = pickle.load(open(synth_path))

    # else:
    if do_shuffle:
        print 'shuffling...'
        real_tup = shuffle(*real_tup, random_state=random_state)
        synth_tup = shuffle(*synth_tup, random_state=random_state)
    
    if scaler_real is not None:
        print 'scaling real data...'
        real_tup = (scaler_real.transform(real_tup[0]), real_tup[1])
        
    if scaler_synth is not None:
        print 'scaling synthetic data...'
        synth_tup = (scaler_synth.transform(synth_tup[0]), synth_tup[1])
        
    if scaler_both is not None:
        print 'scaling real and synthetic data jointly...'
        real_tup = (scaler_both.transform(real_tup[0]), real_tup[1])
        synth_tup = (scaler_both.transform(synth_tup[0]), synth_tup[1])

    if target_scaler is not None:
        print 'scaling targets for both real and synth...'
        real_tup = (real_tup[0], target_scaler.transform(real_tup[1]))
        synth_tup = (synth_tup[0], target_scaler.transform(synth_tup[1]))

        # print 'saving for easy load later...'
        # pickle.dump(real_tup, open(real_path, 'wb'))
        # pickle.dump(synth_tup, open(synth_path, 'wb'))

    n_real = real_tup[0].shape[0]  # total number of obs
    n_synth = synth_tup[0].shape[0]
    
    batch_size_real = int(batch_size * real_to_synth_ratio)
    batch_size_synth = batch_size - batch_size_real
    
    idx_real = np.arange(batch_size_real) % n_real
    idx_synth = np.arange(batch_size_synth) % n_synth
        
    while True:

        # Get real data (features and targets) for this batch.
        Xr = real_tup[0][idx_real]
        Yr = real_tup[1][idx_real]
        idx_real = (idx_real + batch_size_real) % n_real  # vectorized; index cycles so all batches are same size
        
        # Get synthetic data for this batch.
        Xs = synth_tup[0][idx_synth]
        Ys = synth_tup[1][idx_synth]
        idx_synth = (idx_synth + batch_size_synth) % n_synth
        
        # Combine real and synthetic.
        X = np.concatenate((Xr, Xs))
        Y = np.concatenate((Yr, Ys))
        
        if do_shuffle:
            # Shuffle again to mix real with synthetic.
            X, Y = shuffle(X, Y, random_state=random_state)

        if as_tensor:
            X = reshape_as_tensor(X)
        
        yield (X, Y)


class RuntimeHistory(keras.callbacks.Callback):
    '''
    Keras callback to record runtime (wall time) by epoch.
    '''

    def on_train_begin(self, logs={}):
        self.t0 = time.time()
        self.runtime = []
    
    def on_epoch_end(self, batch, logs={}):
        t1 = time.time()
        self.runtime.append(t1 - self.t0)
        self.t0 = t1


def get_max_model_num(continue_from_last_run,
                      dir_models_set):
    '''
    If there was a previous run of this models set, then get the max model
    number. Otherwise just start counting the model at zero ("model_0").
    '''
    if continue_from_last_run:
        # Get max model number.
        try:
            model_files = os.listdir(dir_models_set)
            max_model_num = -1
            for model_file in model_files:
                try:
                    max_model_num = max(int(re.sub('model_', '', model_file)), max_model_num)
                except ValueError:
                    pass
            max_model_num += 1
        except OSError:
            max_model_num = 0
    else:
        max_model_num = 0
        
    return max_model_num        


def app_names_to_filename(app_names):
    return '_'.join([re.sub(' ', '', s) for s in app_names])


def plot_empir_cum(x):
    '''Plot empirical cumulative distribution'''
    # https://stackoverflow.com/questions/15408371/cumulative-distribution-plots-python
    return plt.step(sorted(x), np.arange(len(x))/len(x), color='black')


def get_model_files(path):
    model_files = os.listdir(path)
    model_files = [f for f in model_files if f != '.DS_Store' and os.path.isdir(os.path.join(path, f))]
    return model_files


def random_params():
    return {
        'num_conv_layers': np.random.randint(1, 6),
        'num_dense_layers': np.random.randint(0, 5),
        'start_filters': np.random.choice([2, 4, 8, 16, 32]),
        'deepen_filters': np.random.random() < 0.5,
        'kernel_size': weighted_choice([(3, 1), (6, 1), (12, 1), (24, 1)]),
        'strides': weighted_choice([(1, 1), (2, 1), (3, 1)]),
        'dilation_rate': weighted_choice([(1, 1), (2, 0.13), (3, 0.13)]),
        'do_pool': np.random.random() < 0.5,
        'pool_size': weighted_choice([(2, 1), (4, 1), (8, 1)]),
        'last_dense_layer_size': np.random.choice([8, 16, 32]),
        'dropout_rate_after_conv': np.random.choice([0, 0.1, 0.25, 0.5]),
        'dropout_rate_after_dense': np.random.choice([0, 0.1, 0.25, 0.5]),
        'use_batch_norm': np.random.random() < 0.25,
        'optimizer': np.random.choice([keras.optimizers.Adam,
                                       keras.optimizers.Adagrad,
                                       keras.optimizers.RMSprop]),
        'learning_rate': weighted_choice([(0.01, 0.13),
                                          (0.003, 0.25),
                                          (0.001, 1),
                                          (0.0003, 0.25),
                                          (0.0001, 0.13)]),
        'l2_penalty': weighted_choice([(0, 1), (0.001, 0.25), (0.01, 0.13)])
    }


def plot_errors(history_df, figsize=(11,5), title='Training and validation loss (MSE)'):
    '''
    Plots training and validation errors.
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, title=title) 
    ax.plot(history_df['val_loss'], label='validation loss')
    ax.plot(history_df['loss'], label='training loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    return ax


def extract_targets(all_data, split_type, app_name, app_names, Y_idx):
    '''
    Simply pick the right column of targets.
    '''
    assert split_type in ['train', 'val']
    app_idx = app_names.index(app_name)
    Y = all_data[split_type][Y_idx]
    y = [Y_row[app_idx] for Y_row in Y]
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    return y


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

