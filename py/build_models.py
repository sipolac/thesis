
from __future__ import division

from utils import *
from create_data import *

import os
import pandas as pd
import numpy as np
import re
import shutil

import time
from datetime import datetime, date, timedelta

import cPickle as pickle

import keras
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.utils import plot_model
from keras import regularizers
from keras.utils.generic_utils import get_custom_objects

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib  # need to do this: # need to do this: https://stackoverflow.com/a/21789908/4794432
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

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

    return {
        'X': X,
        'Y1': Y1,
        'Y2': Y2,
        'x_house': x_house,
        'x_date': x_date
    }


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
    
    return {
        'X': X,
        'Y1': Y1,
        'Y2': Y2,
        'x_house': x_house,
        'x_date': x_date
    }


def get_bad_data_tups(dstats, dstats_cond):
    
    if 'Delete_BadAgg' not in dstats.columns:
        dstats['Delete_BadAgg'] = 0
    dstats.loc[dstats_cond, 'Delete_BadAgg'] = 1
    bad_obs_df = dstats.loc[dstats['Delete_BadAgg'] == 1, ['House']]
    bad_obs_df.reset_index(inplace=True)

    dts = bad_obs_df['Time']
    house_ids = bad_obs_df['House']

    tups = []
    for house_id, dt in zip(house_ids, dts):
        tups.append((house_id, dt.date()))

    # dstats.drop('Delete_BadAgg', axis=1, inplace=True)
    # del dstats['Delete_BadAgg']

    return tups


def remove_tups(data_set, tups, is_debug=True):
    '''
    data_set is a list of arrays X, Y1, etc. and tupes is a list of 
    tuples with elements (house_id, datetime). Removes observations
    from arrays where house_id and date(time)s are in the tuple.
    '''
    
    # Find indices 
    bad_idx = []
    for house_id, dt in tups:
        bad_idx.append(np.where(((data_set['x_house'] == house_id) & (data_set['x_date'] == dt)))[0])
    bad_idx = np.concatenate(bad_idx)
    good_idx = [x for x in range(data_set['X'].shape[0]) if x not in bad_idx]

    # Subset each part of data set in list and return new list.
    dat_new = {}
    for key, dat in data_set.iteritems():
        dat_new[key] = dat[good_idx]
    
    if is_debug:
        obs_before = data_set['X'].shape[0]
        obs_after = dat_new['X'].shape[0]
        obs_prop_diff = (obs_before - obs_after) / dat_new['X'].shape[0]
        print 'removed {0} obs ({1:0.2g}% of total)'.format(obs_before - obs_after, obs_prop_diff*100)

    return dat_new


def show_data_dims(all_data, train_dates=None):
    '''
    Debug function for printing out dimensions of all data packet.
    '''
    house_ids = range(1, 22); house_ids.remove(14)  # too annoying to bring in the real array for this

    for split_key in all_data.keys():
        print split_key, type(all_data[split_key])
        for key, dat in all_data[split_key].iteritems():
            print '    {}: {}'.format(key, dat.shape)
        print '    houses: {}'.format(list(set(all_data[split_key]['x_house'])))
        print '    missing houses: {}'.format([h for h in house_ids if h not in list(set(all_data[split_key]['x_house']))])
        print '    num. unique days: {}'.format(len(set(all_data[split_key]['x_date'])))
        if train_dates is not None:
            print '    has training dates: {}'.format(any([d in train_dates for d in list(set(all_data[split_key]['x_date']))]))
            print '    has non-training dates: {}'.format(any([d not in train_dates for d in list(set(all_data[split_key]['x_date']))]))
        print ''


def remove_solar_from_real(all_data, house_ids_solar):
    '''
    Remove data where house has solar panels.
    '''
    assert 'real' in all_data.keys()
    solar_mask = np.in1d(all_data['real']['x_house'], house_ids_solar)
    for key, dat in all_data['real'].iteritems():
        all_data['real'][key] = all_data['real'][key][~solar_mask]
    return all_data


def prepare_real_data(dir_for_model_real,
                      dstats,
                      house_ids_solar,
                      house_ids_not_test_unseen,
                      house_ids_test_unseen,
                      train_dates,
                      all_data = {},
                      is_debug=True):

    '''
    Process real data for model.
    '''

    print 'processing real data...'
    all_data['real'] = load_real_data(dir_for_model_real)  # as list so can alter elements

    print 'removing homes with solar panels...'
    all_data = remove_solar_from_real(all_data, house_ids_solar)

    # Do this after solar to see if it really makes a difference since corr is low w/ solar houses.
    print 'removing obs where correlation between main and sum of apps is low...'
    corr_tups = get_bad_data_tups(dstats, dstats['SumToMainCorr'] < 0.1)
    all_data['real'] = remove_tups(all_data['real'], corr_tups)

    print 'removing obs where agg value is repeated...'
    repeat_cond = dstats[('Appliance0', 'prop_unchanging_large_value')] > 0.1
    corr_tups = get_bad_data_tups(dstats, repeat_cond)
    all_data['real'] = remove_tups(all_data['real'], corr_tups)

    print 'splitting into training, validation/test (seen) and test (unseen)...'
    
    # Split real data into train, val/test for seen houses (same dataset for now), and test for unseen houses.
    masks = {}
    masks['real_train'] = (np.in1d(all_data['real']['x_house'], house_ids_not_test_unseen)) &\
        (np.in1d(all_data['real']['x_date'], train_dates)) 
    masks['val_test_seen'] = (np.in1d(all_data['real']['x_house'], house_ids_not_test_unseen)) &\
        ~(np.in1d(all_data['real']['x_date'], train_dates)) 
    masks['test_unseen'] = np.in1d(all_data['real']['x_house'], house_ids_test_unseen)

    # Print split ratios.
    N_real = all_data['real']['x_house'].shape[0]
    print '-----'
    print '{} training'.format(sum(masks['real_train']) / N_real)
    print '{} validation, seen (combined)'.format(sum(masks['val_test_seen']) / N_real / 2)
    print '{} test, seen (combined)'.format(sum(masks['val_test_seen']) / N_real / 2)  # same as above
    print '{} test, unseen'.format(sum(masks['test_unseen']) / N_real)
    added_bools = masks['real_train'].astype(int) + masks['val_test_seen'].astype(int) + masks['test_unseen'].astype(int)
    if (min(added_bools), max(added_bools)) != (1, 1):
        print 'some obs in split were duplicated or missed!'
    if sum(masks['real_train']) + sum(masks['val_test_seen']) + sum(masks['test_unseen']) != N_real:
        print 'num. obs in in splits don`t equal num. obs in original data'
    print '-----'

    for split_key, mask in masks.iteritems():
        all_data[split_key] = apply_to_dict(lambda x: x[mask], all_data['real'])

    print 'splitting validation/test (seen) into validation and test (seen)...'
    idxs_list = train_test_split(
        range(all_data['val_test_seen']['X'].shape[0]),
        train_size=0.5,  # split 50/50
        stratify=all_data['val_test_seen']['x_house']
    )
    idxs = {}
    idxs['val'] = idxs_list[0]  # index doesn't matter
    idxs['test_seen'] = idxs_list[1]  # index doesn't matter
    for split_key, idx in idxs.iteritems():
        all_data[split_key] = apply_to_dict(lambda x: x[idx], all_data['val_test_seen'])
    
    del all_data['real'], all_data['val_test_seen']

    return all_data


def prepare_synth_data(dir_for_model_synth,
                       all_data = {}):

    print 'loading synthetic data...'
    all_data['synth_train_all'] = load_synth_data(dir_for_model_synth)

    # print 'datasets: {}'.format(all_data.keys())
        
    return all_data


def create_scalers(all_data):
    '''
    Create scaler/standardizer for real data and synthetic data, and then
    both when they're combined.
    '''
    # Create scalers for real and synthetic (mean only since want to preserve differences between
    # data points).
    scaler_real = StandardScaler(with_std=False).fit(all_data['real_train']['X'].reshape(-1,1))
    scaler_synth = StandardScaler(with_std=False).fit(all_data['synth_train_all']['X'].reshape(-1,1))

    # Create scaler for combined real and synthetic (standard dev only
    # since data was already demeaned separately). Sample synthetic
    # so that there are the same number of obs for real and synthetic.
    n_train_real = all_data['real_train']['X'].shape[0] 
    sample_idx = np.random.choice(all_data['synth_train_all']['X'].shape[0], n_train_real, replace=False)
    X_train = np.concatenate((all_data['synth_train_all']['X'][sample_idx],
                             all_data['real_train']['X']))
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


# def print_layer_shapes(model):
#     shape_pattern = 'shape=(.+?), dtype'
#     for layer in model.layers:
#         print '{}: {}'.format(layer.name, re.search(shape_pattern, str(layer.output)).group(1))


def create_model(
    num_outputs,
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
    l2_penalty,
    hidden_layer_activation,
    output_layer_activation,
    loss
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
    if K.image_data_format() == 'channels_last':
        input_shape = (n_per_day, 1)
    else:
        input_shape = (1, n_per_day)

    model = Sequential()
    
    # Add convolutional layers.
    for layer_num in range(num_conv_layers):
        
        filter_multiplier = 2**layer_num if deepen_filters else 1
        filters = start_filters * filter_multiplier
        
        conv_args = {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
            'padding': 'same',
            'dilation_rate': dilation_rate,
            'activation': hidden_layer_activation,
            'name': 'conv_{}'.format(layer_num),
            'kernel_regularizer': kernel_regularizer}
        if layer_num == 0:
            # Need input shape if first layer.
            conv_args['input_shape'] = input_shape
        
        model.add(Conv1D(**conv_args))
        
        if do_pool:
            model.add(MaxPooling1D(
                pool_size,
                padding='same',
                name='pool_{}'.format(layer_num)))
    
    model.add(Dropout(dropout_rate_after_conv, name='dropout_after_conv'))
    model.add(Flatten(name='flatten'))
    
    # Add dense layers.
    for layer_num in range(num_dense_layers):
        layer_size = last_dense_layer_size * 2**(num_dense_layers - layer_num - 1)
        model.add(Dense(
            layer_size,
            activation=hidden_layer_activation,
            kernel_regularizer=kernel_regularizer,
            name='dense_{}'.format(layer_num)))
        model.add(Dropout(
            dropout_rate_after_dense,
            name='dropout_dense_{}'.format(layer_num)))
        if use_batch_norm:
            model.add(BatchNormalization(name='batch_norm_{}'.format(layer_num)))
    
    model.add(Dense(
        num_outputs,
        activation=output_layer_activation,
        name='dense_output',
        kernel_regularizer=kernel_regularizer))
    
    model.compile(loss=loss, optimizer=optimizer(lr=learning_rate))
    
    return model


def create_target_scaler(all_data, Y_key, app_idx):
    target_scaler = StandardScaler(with_mean=False).fit(
        np.concatenate((all_data['synth_train_all'][Y_key][:,app_idx],
                        all_data['real_train'][Y_key][:,app_idx])))
    return target_scaler


def run_models(
    all_data,
    target_type,  # 'energy' or 'activations'
    app_names,
    APP_NAMES,
    dir_models,
    params_function,
    modeling_group_name = str(date.today()),
    models_to_run = 10,
    epochs = 100,
    batch_size = 32,
    continue_from_last_run = True,
    total_obs_per_epoch = 8192,
    real_to_synth_ratio = 0.5,
    patience = 6,
    checkpointer_verbose = 0,
    fit_verbose = 1,
    show_plot = False,
    print_summary = True,
    model_init_dir = None,  # not really initialization; just taking params
    model_num_stop = None
):
    
    assert target_type in ['energy', 'activations']
    
    if isinstance(app_names, basestring):
        app_names = [app_names]

    steps_per_epoch = total_obs_per_epoch // batch_size
    dir_models_set = os.path.join(dir_models, modeling_group_name, target_type, app_names_to_filename(app_names))
    Y_key = 'Y1' if target_type=='energy' else 'Y2'  # choose which targets to use

    # "Append" results to last runs of models, if they exist?
    max_model_num = get_max_model_num(continue_from_last_run, dir_models_set)
    if model_num_stop is not None and max_model_num >= model_num_stop:
        print 'no models to run!'
        return None

    # Get the index in the Y array of the target appliance(s).
    app_idx = []
    for app_name in app_names:
        app_idx.append(APP_NAMES.index(app_name))
    
    # print 'removing extreme values...'
    # for Y_key in ['Y1', 'Y2']:
    #     all_data['synth_train_all'] = remove_extremes(all_data['synth_train_all'], extreme_percentile_cutoff, Y_key

    # Create scaler for the Y variable.
    target_scaler = create_target_scaler(all_data, Y_key, app_idx)

    real_tup = (all_data['real_train']['X'], all_data['real_train'][Y_key][:,app_idx])
    synth_tup = (all_data['synth_train_all']['X'], all_data['synth_train_all'][Y_key][:,app_idx])
    generator = generate_data(real_tup,
                              synth_tup,
                              scaler_real, scaler_synth, scaler_both, target_scaler,
                              batch_size = batch_size,
                              real_to_synth_ratio = real_to_synth_ratio)

    for model_num in np.arange(models_to_run) + max_model_num:

        if model_num_stop is not None and model_num >= model_num_stop:
            print 'no more models to run!'
            return None

        model_name = 'model_{}'.format(model_num)
        # model_name = 'simple_small'
        
        if model_init_dir is None:
            params = params_function()
        else:
            params = pickle.load(open(os.path.join(model_init_dir, 'params.pkl'), 'rb'))
        
        print '='*25 + '\n{}\n'.format(model_name) + '='*25
        print pd.DataFrame.from_dict(params, orient='index')
        print '\n' + '='*25 + '\n'

        model = create_model(len(app_names), N_PER_DAY, **params)
        if print_summary:
            print model.summary()

        dir_this_model = os.path.join(dir_models_set, model_name)
        model_filename = os.path.join(dir_this_model, 'weights.hdf5')
        history_filename = os.path.join(dir_this_model, 'history.csv')
        params_csv_filename = os.path.join(dir_this_model, 'params.csv')
        params_pkl_filename = os.path.join(dir_this_model, 'params.pkl')
        makedirs2(dir_this_model)
        
        try:

            # Save target scaler to recover targets later. Save for every model just in case
            # data changes or something (even though if it doesn't change you can use the same
            # target scaler for all models for this target type / app names / modeling group
            # combination)
            pickle.dump(target_scaler, open(os.path.join(dir_this_model, 'target_scaler.pkl'), 'wb'))

            # Save params in two forms: CSV for easy opening w/ Excel, and pickled to
            # preserve data in correct format (otherwise all param values are converted
            # into strings).
            pickle.dump(params, open(params_pkl_filename, 'wb'))
            pd.DataFrame.from_dict(params, orient='index').to_csv(params_csv_filename, header=False)

            # Define callbacks
            # https://keras.io/callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
            csvlogger = CSVLogger(history_filename, separator=',', append=False)
            checkpointer = ModelCheckpoint(filepath=model_filename, verbose=checkpointer_verbose, save_best_only=True)
            runtime_history = RuntimeHistory()

            X_val_fit = reshape_as_tensor(all_data['val']['X'])
            Y_val_fit = target_scaler.transform(all_data['val'][Y_key][:,app_idx])
            history = model.fit_generator(generator,
                                          steps_per_epoch = steps_per_epoch,
                                          epochs = epochs,
                                          validation_data=(X_val_fit, Y_val_fit),
                                          callbacks = [early_stopping, csvlogger, checkpointer, runtime_history],
                                          verbose = fit_verbose)

        except KeyboardInterrupt:
            print 'removing dir due to keyboard interrupt: {}'.format(dir_this_model)
            shutil.rmtree(dir_this_model)
            raise KeyboardInterrupt

        # Add runtimes to CSV.
        history_df = pd.read_csv(history_filename)
        history_df['runtime'] = runtime_history.runtime
        history_df.to_csv(history_filename, index=False)

        # # Load best model.
        # model_filename = os.path.join(dir_this_model, 'weights.hdf5')
        # model = load_model(model_filename)
        
        if show_plot:
            plot_errors(history_df)
            plt.show()


def get_extreme_mask(X, q):
    '''
    Outputs mask that returns obs w/ extreme values for any of the features of array X.
    Note the broadcasting in the function: each feature is treated separately.
    '''
    extreme_mask = np.any(X > np.percentile(X, q=q, axis=0), axis=1)
    return extreme_mask


def remove_extremes(data_set, q, Y_key):
    '''
    Takes data_set (list of arrays X, Y, x_house, ...) and removes obs where values in
    data as identified by Y_day are extreme (according to cutoff q). Note that this
    an observation is removed if *any* column for any appliance is extreme.
    '''
    extreme_mask = get_extreme_mask(data_set[Y_key], q=q)
    new_data = {}
    for key, dat in data_set.iteritems():
        new_data[key] = dat[~extreme_mask]
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
    if app_names == ['fridge', 'kettle', 'washing machine', 'dishwasher', 'microwave']:  # cheating a bit, but a pain to bring in APP_NAMES
        app_names = ['all_apps']
    if isinstance(app_names, basestring):
        app_names = [app_names]
    return '_'.join([re.sub(' ', '', s) for s in app_names])


def get_model_files(path):
    model_files = os.listdir(path)
    model_files = [f for f in model_files if f != '.DS_Store' and os.path.isdir(os.path.join(path, f))]
    return model_files


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


def get_histories_df(dir_models_set):

    model_files = get_model_files(dir_models_set)

    history_all = []
    params_all = []

    for model_name in model_files:
        
        try:
            history_df = pd.read_csv(os.path.join(dir_models_set, model_name, 'history.csv'))
        except:
            print '{} has no history data'.format(model_name)
            print 'dir_models_set: {}'.format(dir_models_set)
        param_colnames = ['param', 'value']
        try:
            params = pickle.load(open(os.path.join(dir_models_set, model_name, 'params.pkl'), 'rb'))
            params_df = pd.DataFrame.from_dict(params, orient='index').reset_index()
            params_df.columns = param_colnames
        except IOError:
            # warnings.warn(str(e))
            # In case it's before I started pickling parameters as dictionaries.
            params_df = pd.read_csv(os.path.join(dir_models_set, model_name, 'params.csv'),
                                    header=None,
                                    names=param_colnames)
        
        history_df['model'] = model_name
        try:
            history_df['runtime']
        except KeyError:
            history_df['runtime'] = float('inf')
        params_df['model'] = model_name    

        history_all.append(history_df)
        params_all.append(params_df)

        # print model_name
        # print params_df

        # plot_errors(history_df, figsize=(11,5))
        # plt.show()

    params_all = pd.concat(params_all)
    history_all = pd.concat(history_all)

    # print all_history
    params_wide = params_all.pivot(index='model', columns='param')
    params_wide.rename(columns={'loss': 'loss_fn'}, inplace=True)  # to make sure there aren't two loss columns
    history_best = history_all.groupby('model').agg({
        'loss': min,
        'val_loss': min,
        'runtime': min,
        'epoch': len
    })

    hist_and_params = history_best.join(params_wide)
    
    # Remove "value" element (i.e., first level of column multiindex heirarchy)
    # in column tuples ("value", [column]).
    remove_first = lambda x: x[-1] if not isinstance(x, basestring) else x
    hist_and_params.columns = [remove_first(x) for x in hist_and_params.columns]

    hist_and_params.sort_values('val_loss', inplace=True)
    # hist_and_params.to_csv('tmp.csv')

    return hist_and_params


def get_best_model_name(dir_models_set):
    hist_and_params = get_histories_df(dir_models_set)
    best_model_name = hist_and_params.loc[hist_and_params['val_loss']==hist_and_params['val_loss'].min()].index.values[0]
    return best_model_name

def load_best_model(dir_models_set):
    best_model_name = get_best_model_name(dir_models_set)
    dir_best_model = os.path.join(dir_models_set, best_model_name)
    return load_model(os.path.join(dir_best_model, 'weights.hdf5'))


def plot_series_activations(
    house_id,
    dt,
    activations,
    apps_to_plot=None,
    figsize=(8,5.5),
    height_ratios=[5, 1],
    cmap_series=plt.cm.tab10,
    cmap_heatmap='Greys'
):
    
    # Define axes.
    fig = plt.figure(figsize=figsize)
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=height_ratios)
    ax_series = fig.add_subplot(gs[0])
    ax_heatmap = fig.add_subplot(gs[1])
    
    # Plot series.
    plot_day(house_id, dt=dt, apps_to_plot=apps_to_plot, ax=ax_series, cmap=cmap_series)
    ax_series.get_xaxis().set_ticks_position('top')

    # Plot heatmap of activations.
    sns.heatmap(activations, ax=ax_heatmap, cbar=False, cmap=cmap_heatmap)
    ax_heatmap.get_xaxis().set_visible(False)
    ax_heatmap.set_ylabel('Last conv.\nlayer\nactivations')
    ax_heatmap.set_yticklabels([])

    gs.tight_layout(fig)
    plt.show()
    
    return gs


def plot_pred_scatter(
    y,
    y_hat,
    x_house,
    house_ids_not_test_unseen,
    file_suffix,
    save_dir = None,
    title = None,
    lowess = False,
    fit_reg = True,
    palette = np.array(sns.color_palette('tab10', 10))[[0,1]],  # gets blue and orange from tab10 palette
):

    plt.close()
    
    max_val = max([max(y), max(y_hat)])
    # xy_line = plt.plot([0,max_val], [0,max_val], ':', color='gray')

    if save_dir is not None:
        plot_dir = os.path.join(dir_run)
        makedirs2(plot_dir)

    # Create dataset for plotting.
    plot_data = pd.DataFrame.from_dict({'Actual': y, 'Predicted': y_hat, 'House': x_house})
    plot_data['House type'] = 'unseen'
    plot_data.loc[plot_data['House'].isin(house_ids_not_test_unseen), 'House type'] = 'seen'

    plot_args = {
        'x': 'Actual',
        'y': 'Predicted',
        'data': plot_data,
        'lowess': lowess,
        'truncate': True,
        'fit_reg': fit_reg
    }
    
    g = sns.lmplot(scatter_kws={'color': 'black', 'alpha': 0.3},
                   line_kws={'color': 'black'},
                   **plot_args
    )
    lims = [-max_val*0.05,max_val*1.1]
    g.set(ylim=lims, xlim=lims)
    plt.subplots_adjust(top=0.83)
    g.fig.suptitle(title, size=14)
    if save_dir is not None:
        filename = '{}_{}_{}.pdf'.format(
            'predscatter',
            target_type,
            app_names_to_filename(app_names)
        )
        plt.savefig(os.path.join(plot_dir, filename))
    plt.show()

    g = sns.lmplot(scatter_kws={'alpha': 0.3},
                   col='House',
                   hue='House type',
                   col_wrap=4,
                   size=2.3,
                   aspect=0.85,
                   palette=palette,
                   **plot_args
    )
    # plt.ylim(max_val*1.1)
    # plt.xlim(max_val*1.1)
    # g = g.map(plt.plot, [0,max_val], [0,max_val], ':', color='gray')
    g.set(ylim=lims, xlim=lims)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle(title, size=14)
    if save_dir is not None:
        filename = '{}_{}_{}_{}.pdf'.format(
            'predscatterbyhouse',
            target_type,
            app_names_to_filename(app_names),
            file_suffix)
        plt.savefig(os.path.join(plot_dir, filename))
    plt.show()
        
        
def truncate_model(model, conv_layer_index=-1):
    '''
    Truncate model at certain convolutional layer. conv_layer_index is the
    index of the convoluational layer (0 is the first, -1 is the last)
    '''
    layer_names = [x.name for x in model.layers]
    conv_layer_name = [l for l in layer_names if 'conv_' in l][conv_layer_index]
    model_truncated = Sequential()
    for layer in model.layers:
        model_truncated.add(layer)
        if layer.name == conv_layer_name:
            break
    return model_truncated


def get_activations(x, truncated_model):
    '''
    Get activations of hidden layer from truncated model.
    '''
    return truncated_model.predict(reshape_as_tensor([x]))[0].T


def reduce_dims_activations(activations, k): 
    '''
    Use PCA to reduce dimensionality of activations. activations is 
    the N x [|filters|] matrix and k is number of output dimensions.
    '''
    if activations.shape[0] < k:
        return activations
    activations_pca = PCA(k).fit_transform(activations.T).T
    return activations_pca - activations_pca.min()  # scale so min is zero


if __name__ == '__main__':

    dir_proj = '/Users/sipola/Google Drive/education/coursework/graduate/edinburgh/dissertation/thesis'
    dir_data = os.path.join(dir_proj, 'data')
    dir_for_model = os.path.join(dir_data, 'for_model')
    dir_for_model_real = os.path.join(dir_for_model, 'real')
    dir_for_model_synth = os.path.join(dir_for_model, 'synthetic')
    dir_models = os.path.join(dir_data, 'models')
    dir_run = os.path.join(dir_proj, 'run', str(date.today()))
    path_daily_stats = os.path.join(dir_data, 'stats_by_day.pkl')

    N_PER_DAY = 14400  # 24 * 60 * 60 / 6
    HOUSE_IDS, HOUSE_IDS_TEST_UNSEEN, HOUSE_IDS_NOT_TEST_UNSEEN, HOUSE_IDS_SOLAR = get_house_id_groups()
    HOUSE_IDS_NOT_SOLAR = [h for h in HOUSE_IDS if h not in HOUSE_IDS_SOLAR]
    # TRAIN_VAL_DATE_MAX = date(2015,2,28)
    APP_NAMES = ['fridge', 'kettle', 'washing machine', 'dishwasher', 'microwave']
    TRAIN_DTS = np.load(os.path.join(dir_for_model_synth, 'train_dts.npy'))

    take_diff = False
    # val_prop = 0.2
    train_dates = [dt.date() for dt in TRAIN_DTS]
    extreme_percentile_cutoff = 100

    # Set random seed so training/validation set is consistent across runs.
    np.random.seed(20170627)

    dstats = pd.read_pickle(path_daily_stats)
    dstats = clean_daily_stats(dstats, is_debug=False)

    all_data = prepare_real_data(dir_for_model_real,
                                 dstats,
                                 HOUSE_IDS_SOLAR,
                                 HOUSE_IDS_NOT_TEST_UNSEEN,
                                 HOUSE_IDS_TEST_UNSEEN,
                                 train_dates)

    all_data = prepare_synth_data(dir_for_model_synth,
                                  all_data = all_data)
    show_data_dims(all_data, train_dates)

    # Want to take diffs before making scalers.
    if take_diff:
        print 'taking diffs...'
        for key, dat in all_data.iteritems():
            all_data[key]['X'] = take_diff_df(dat['X'])

    print 'creating scalers...'
    scaler_real, scaler_synth, scaler_both = create_scalers(all_data)

    print 'scaling validation and test data...'
    for split_type in ['val', 'test_seen', 'test_unseen']:
        all_data[split_type]['X'] = scaler_real.transform(all_data[split_type]['X'])
        all_data[split_type]['X'] = scaler_both.transform(all_data[split_type]['X'])

    # Now set random seed to something specific to this run so that hyperparameters
    # are different each time this is run.
    np.random.seed(int(time.time()))

    real_deal = True
    if real_deal:

        today = str(date.today())
        modeling_group_name = 'main'
        # modeling_group_name = today

        def random_params():
            return {
                # 'num_conv_layers': np.random.randint(3, 8),
                'num_conv_layers': np.random.randint(4, 8),
                'num_dense_layers': np.random.randint(1, 3),
                'start_filters': int(rand_geom(4, 9)),
                'deepen_filters': True,
                'kernel_size': int(rand_geom(3, 7)),
                'strides': int(rand_geom(1, 3)),
                'dilation_rate': 1,
                'do_pool': True,
                'pool_size': int(rand_geom(2, 5)),
                'last_dense_layer_size': int(rand_geom(8, 32)),
                'dropout_rate_after_conv': 0.5,
                'dropout_rate_after_dense': 0.25,
                'use_batch_norm': False,
                'optimizer': keras.optimizers.Adam,
                'learning_rate': rand_geom(0.0003, 0.003),
                'l2_penalty': np.random.choice([0, rand_geom(0.00000001, 0.000001)]),
                'hidden_layer_activation': 'relu',
                'output_layer_activation': 'relu',
                'loss': 'mse'
            }

        for _ in range(10):

            print 'starting modeling loops...'

            for target_type in ['energy', 'activations']:
                # for app_names in shuffle(APP_NAMES + [APP_NAMES]):  # each element can be a list or a basestring
                for app_names in APP_NAMES + [APP_NAMES]:

                    print '\n\n' + '*'*25
                    print 'target variable: {}'.format(target_type)
                    print 'target appliance(s): {}'.format(app_names)
                    print '*'*25 + '\n\n'

                    run_models(
                        all_data,
                        target_type,
                        app_names,
                        APP_NAMES,
                        dir_models,
                        params_function = random_params,
                        modeling_group_name = modeling_group_name,
                        models_to_run = 3,
                        epochs = 100,
                        batch_size = 32,
                        continue_from_last_run = True,
                        total_obs_per_epoch = 8192,
                        real_to_synth_ratio = 0.5,
                        patience = 5,
                        checkpointer_verbose = 0,
                        fit_verbose = 1,
                        show_plot = False,
                        model_num_stop = 20)

    else:
        def static_params1():
            return {
            'num_conv_layers': 7,
            'num_dense_layers': 3,
            'start_filters': 2,
            'deepen_filters': True,
            'kernel_size': 3,
            'strides': 2,
            'dilation_rate': 1,
            'do_pool': True,
            'pool_size': 2,
            'last_dense_layer_size': 8,
            'dropout_rate_after_conv': 0.5,
            'dropout_rate_after_dense': 0.25,
            'use_batch_norm': False,
            'optimizer': keras.optimizers.Adam,
            'learning_rate': 0.001,
            'l2_penalty': 0,
            'hidden_layer_activation': 'relu',
            'output_layer_activation': 'relu',
            'loss': 'mse'
            }

        # def static_params2():
        #     return {
        #     'num_conv_layers': 5,
        #     'num_dense_layers': 2,
        #     'start_filters': 4,
        #     'deepen_filters': True,
        #     'kernel_size': 3,
        #     'strides': 2,
        #     'dilation_rate': 1,
        #     'do_pool': True,
        #     'pool_size': 4,
        #     'last_dense_layer_size': 8,
        #     'dropout_rate_after_conv': 0.5,
        #     'dropout_rate_after_dense': 0.25,
        #     'use_batch_norm': False,
        #     'optimizer': keras.optimizers.Adam,
        #     'learning_rate': 0.001,
        #     'l2_penalty': 0,
        #     'hidden_layer_activation': 'relu',
        #     'output_layer_activation': 'relu',
        #     'loss': 'mse'
        #     }

        # def random_params():
        #     return {
        #         'num_conv_layers': np.random.randint(1, 6),
        #         'num_dense_layers': np.random.randint(0, 5),
        #         'start_filters': np.random.choice([2, 4, 8, 16, 32]),
        #         'deepen_filters': np.random.random() < 0.5,
        #         'kernel_size': weighted_choice([(3, 1), (6, 1), (12, 1), (24, 1)]),
        #         'strides': weighted_choice([(1, 1), (2, 1), (3, 1)]),
        #         'dilation_rate': weighted_choice([(1, 1), (2, 0.13), (3, 0.13)]),
        #         'do_pool': np.random.random() < 0.5,
        #         'pool_size': weighted_choice([(2, 1), (4, 1), (8, 1)]),
        #         'last_dense_layer_size': np.random.choice([8, 16, 32]),
        #         'dropout_rate_after_conv': np.random.choice([0, 0.1, 0.25, 0.5]),
        #         'dropout_rate_after_dense': np.random.choice([0, 0.1, 0.25, 0.5]),
        #         'use_batch_norm': np.random.random() < 0.25,
        #         'optimizer': np.random.choice([keras.optimizers.Adam,
        #                                        keras.optimizers.Adagrad,
        #                                        keras.optimizers.RMSprop]),
        #         'learning_rate': weighted_choice([(0.01, 0.13),
        #                                           (0.003, 0.25),
        #                                           (0.001, 1),
        #                                           (0.0003, 0.25),
        #                                           (0.0001, 0.13)]),
        #         'l2_penalty': weighted_choice([(0, 1), (0.001, 0.25), (0.01, 0.13)])
        #     }

        run_models(
            all_data,
            'energy',  # 'energy' or 'activations'
            'washing machine',
            APP_NAMES,
            dir_models,
            params_function = static_params1,
            modeling_group_name = '2017-07-02',
            models_to_run = 1,
            epochs = 100,
            batch_size = 32,
            continue_from_last_run = True,
            total_obs_per_epoch = 8192,
            real_to_synth_ratio = 0.5,
            patience = 5,
            checkpointer_verbose = 0,
            fit_verbose = 1,
            show_plot = False)

        # run_models(
        #     all_data,
        #     'energy',  # 'energy' or 'activations'
        #     'washing machine',
        #     APP_NAMES,
        #     dir_models,
        #     params_function = static_params2,
        #     modeling_group_name = '2017-07-02',
        #     models_to_run = 1,
        #     epochs = 100,
        #     batch_size = 32,
        #     continue_from_last_run = True,
        #     total_obs_per_epoch = 8192,
        #     real_to_synth_ratio = 0.5,
        #     patience = 5,
        #     checkpointer_verbose = 0,
        #     fit_verbose = 1,
        #     show_plot = False)

    # 