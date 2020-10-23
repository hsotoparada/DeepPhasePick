#module util_dpp, for python 3.x
#coding=utf-8

import obspy.core as oc
import numpy as np
from glob import glob
from obspy.signal.trigger import trigger_onset
from obspy.io.mseed.core import InternalMSEEDError
import matplotlib as mpl
import matplotlib.ticker as ticker
import pylab as plt
from datetime import datetime
from keras.models import load_model
from keras import optimizers, utils
import keras
import tensorflow as tf
from hyperas.utils import eval_hyperopt_space
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import tqdm
from operator import itemgetter
import re, sys, os, shutil, gc
import pickle


##  FUNCTIONS  ##

def export_dict2pckl(dct, opath):
    """
    export dictionary as pickle file.
    dct = dictionary
    opath = output path to created pickle file.
    """
    with open(out_path, 'wb') as pout:
        pickle.dump(dct, pout)


def import_pckl2dict(ipath):
    """
    import pickle file to dictionary. It returns this dictionary.
    ipath = path to pickle file.
    """
    with open(ipath, 'rb') as pin:
        dct = pickle.load(pin)
    return dct


def init_session():
    """
    sets up tensorflow v2.x / keras session
    """
    #
    # This is to avoid error:
    # Failed to get convolution algorithm. This is probably because cuDNN failed to initialize...
    config = tf.compat.v1.ConfigProto()
    # config = tf.ConfigProto()
    #
    # It allows any new GPU process which consumes a GPU memory to be run on the same machine.
    # see: https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # tf.keras.backend.set_session(tf.Session(config=config))
    session = tf.compat.v1.Session(config=config)
    # session = tf.Session(config=config)
    #
    # remove previously generated files or directories
    dirs_remove = ['__pycache__/', '~/.nv/']
    for dir_remove in dirs_remove:
        try:
            shutil.rmtree(dir_remove)
            print(f"{dir_remove} removed")
        except FileNotFoundError:
            print(f"{dir_remove} not found, continuing...")
            pass


def get_model_detection(ipath, ntrials, verbose=True):
    """
    retrieves dictionary with best model and other relevant results obtained from
    hyperparameter optimization performed for phase detection task
    """
    #
    space = get_space_det()
    trials = import_pckl2dict(f"{ipath}/trials_hyperopt_ntrials_{ntrials:03}.pckl")
    arg_best_trial = get_arg_best_trial(trials)
    best_trial = trials.trials[arg_best_trial]
    best_model = load_model(f"{ipath}/model_hyperopt_t{arg_best_trial:03}.h5")
    best_params = eval_hyperopt_space(space, best_trial['misc']['vals'])
    best_results = import_pckl2dict(f"{ipath}/dict_hyperopt_t{arg_best_trial:003}.pckl")
    best_hist = best_results['history']
    #
    if verbose:
        print("#")
        print(best_model.summary())
        #
        print("######")
        print(f"best model for phase detection found for trial {arg_best_trial:03}/{ntrials:003} and hyperparameters:")
        for k in best_params.keys():
            print(k, best_params[k])
        print("#")
        print(f"best loss for phase detection:")
        print(np.array(best_hist['val_acc']).max())
    #
    dct = {
        'best_model': best_model,
        'best_params': best_params,
        'best_results': best_results,
        'best_hist': best_hist,
    }
    #
    return dct


def get_model_picking(ipath, mode, ntrials, verbose=True):
    """
    retrieves dictionary with best model and other relevant results obtained from
    hyperparameter optimization performed for phase picking task
    """
    #
    if mode == 'P':
        space = get_space_pick_P()
    else:
        space = get_space_pick_S()
    #
    trials = import_pckl2dict(f"{ipath}/trials_hyperopt_ntrials_{ntrials:03}.pckl")
    arg_best_trial = get_arg_best_trial(trials)
    best_trial = trials.trials[arg_best_trial]
    best_model = load_model(f"{ipath}/model_hyperopt_t{arg_best_trial:03}.h5")
    best_params = eval_hyperopt_space(space, best_trial['misc']['vals'])
    best_results = import_pckl2dict(f"{ipath}/dict_hyperopt_t{arg_best_trial:03}.pckl")
    best_hist = best_results['history']
    #
    if verbose:
        print("#")
        print(best_model.summary())
        #
        print("######")
        print(f"best model for {mode} phase picking found for trial {arg_best_trial:03}/{ntrials:03} and hyperparameters:")
        for k in best_params.keys():
            print(k, best_params[k])
        print("#")
        print(f"best val acc for {mode} phase picking:")
        print(np.array(best_hist['val_acc']).max())
    #
    dct = {
        'best_model': best_model,
        'best_params': best_params,
        'best_results': best_results,
        'best_hist': best_hist,
    }
    #
    return dct


#TODO: add reference to source code (Ross et al., 2018)
def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )
    #
    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )
    #
    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )
    #
    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)
    #
    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])
    #
    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )
    #
    if copy:
        return strided.copy()
    else:
        return strided


def get_arg_best_trial(trials):
    """
    retrieves index of best trial (trial at which loss in minimum).
    """
    losses = [float(trial['result']['loss']) for trial in trials]
    arg_min_loss = np.argmin(losses)
    return arg_min_loss


def get_space_det():
    """
    returns hyperopt search space used in hyperparameter optimization of model trained for the phase detection task
    """
    #
    space = {
		#
		# convolutional layers
		#
		'conv_knsize_1': hp.choice('conv_knsize_1', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
		'conv_activ_1': hp.choice('conv_activ_1', ['relu', 'sigmoid']),
		'conv_drop_1': hp.choice('conv_drop_1', np.arange(.2, .51, .05)),
		# 'conv_filters': hp.choice('conv_filters', [4, 8, 16, 32]),
		'conv_filters': hp.choice('conv_filters', np.arange(2, 33, 2)),
		'conv_nlay': hp.choice('conv_nlay', [
			{'layers': 'one'},
			{'layers': 'two',
			 'conv_knsize_2_1': hp.choice('conv_knsize_2_1', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_2_1': hp.choice('conv_activ_2_1', ['relu', 'sigmoid']),
			 'conv_drop_2_1': hp.choice('conv_drop_2_1', np.arange(.2, .51, .05)),
			 'conv_augfilter_2_1': hp.choice('conv_augfilter_2_1', [1, 2]),
			 },
			{'layers': 'three',
			 'conv_knsize_3_1': hp.choice('conv_knsize_3_1', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_3_1': hp.choice('conv_activ_3_1', ['relu', 'sigmoid']),
			 'conv_drop_3_1': hp.choice('conv_drop_3_1', np.arange(.2, .51, .05)),
			 'conv_augfilter_3_1': hp.choice('conv_augfilter_3_1', [1, 2]),
			 'conv_knsize_3_2': hp.choice('conv_knsize_3_2', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_3_2': hp.choice('conv_activ_3_2', ['relu', 'sigmoid']),
			 'conv_drop_3_2': hp.choice('conv_drop_3_2', np.arange(.2, .51, .05)),
			 'conv_augfilter_3_2': hp.choice('conv_augfilter_3_2', [1, 2]),
			 },
			{'layers': 'four',
			 'conv_knsize_4_1': hp.choice('conv_knsize_4_1', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_4_1': hp.choice('conv_activ_4_1', ['relu', 'sigmoid']),
			 'conv_drop_4_1': hp.choice('conv_drop_4_1', np.arange(.2, .51, .05)),
			 'conv_augfilter_4_1': hp.choice('conv_augfilter_4_1', [1, 2]),
			 'conv_knsize_4_2': hp.choice('conv_knsize_4_2', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_4_2': hp.choice('conv_activ_4_2', ['relu', 'sigmoid']),
			 'conv_drop_4_2': hp.choice('conv_drop_4_2', np.arange(.2, .51, .05)),
			 'conv_augfilter_4_2': hp.choice('conv_augfilter_4_2', [1, 2]),
			 'conv_knsize_4_3': hp.choice('conv_knsize_4_3', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_4_3': hp.choice('conv_activ_4_3', ['relu', 'sigmoid']),
			 'conv_drop_4_3': hp.choice('conv_drop_4_3', np.arange(.2, .51, .05)),
			 'conv_augfilter_4_3': hp.choice('conv_augfilter_4_3', [1, 2]),
			 },
			{'layers': 'five',
			 'conv_knsize_5_1': hp.choice('conv_knsize_5_1', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_5_1': hp.choice('conv_activ_5_1', ['relu', 'sigmoid']),
			 'conv_drop_5_1': hp.choice('conv_drop_5_1', np.arange(.2, .51, .05)),
			 'conv_augfilter_5_1': hp.choice('conv_augfilter_5_1', [1, 2]),
			 'conv_knsize_5_2': hp.choice('conv_knsize_5_2', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_5_2': hp.choice('conv_activ_5_2', ['relu', 'sigmoid']),
			 'conv_drop_5_2': hp.choice('conv_drop_5_2', np.arange(.2, .51, .05)),
			 'conv_augfilter_5_2': hp.choice('conv_augfilter_5_2', [1, 2]),
			 'conv_knsize_5_3': hp.choice('conv_knsize_5_3', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_5_3': hp.choice('conv_activ_5_3', ['relu', 'sigmoid']),
			 'conv_drop_5_3': hp.choice('conv_drop_5_3', np.arange(.2, .51, .05)),
			 'conv_augfilter_5_3': hp.choice('conv_augfilter_5_3', [1, 2]),
			 'conv_knsize_5_4': hp.choice('conv_knsize_5_4', [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
			 'conv_activ_5_4': hp.choice('conv_activ_5_4', ['relu', 'sigmoid']),
			 'conv_drop_5_4': hp.choice('conv_drop_5_4', np.arange(.2, .51, .05)),
			 'conv_augfilter_5_4': hp.choice('conv_augfilter_5_4', [1, 2]),
			 }
		]),
		#
		# dense layers
		#
		'dense_units_1': hp.choice('dense_units_1', np.arange(25, 301, 25)),
		'dense_activ_1': hp.choice('dense_activ_1', ['relu', 'sigmoid']),
		'dense_drop_1': hp.choice('dense_drop_1', np.arange(.2, .51, .05)),
		'dense_nlay': hp.choice('dense_nlay', [
			{'layers': 'one'},
			{'layers': 'two',
			 'dense_units_2_1': hp.choice('dense_units_2_1', np.arange(25, 301, 25)),
			 'dense_activ_2_1': hp.choice('dense_activ_2_1', ['relu', 'sigmoid']),
			 'dense_drop_2_1': hp.choice('dense_drop_2_1', np.arange(.2, .51, .05)),
			 },
			{'layers': 'three',
			 'dense_units_3_1': hp.choice('dense_units_3_1', np.arange(25, 301, 25)),
			 'dense_activ_3_1': hp.choice('dense_activ_3_1', ['relu', 'sigmoid']),
			 'dense_drop_3_1': hp.choice('dense_drop_3_1', np.arange(.2, .51, .05)),
			 'dense_units_3_2': hp.choice('dense_units_3_2', np.arange(25, 301, 25)),
			 'dense_activ_3_2': hp.choice('dense_activ_3_2', ['relu', 'sigmoid']),
			 'dense_drop_3_2': hp.choice('dense_drop_3_2', np.arange(.2, .51, .05)),
			 },
			{'layers': 'four',
			 'dense_units_4_1': hp.choice('dense_units_4_1', np.arange(25, 301, 25)),
			 'dense_activ_4_1': hp.choice('dense_activ_4_1', ['relu', 'sigmoid']),
			 'dense_drop_4_1': hp.choice('dense_drop_4_1', np.arange(.2, .51, .05)),
			 'dense_units_4_2': hp.choice('dense_units_4_2', np.arange(25, 301, 25)),
			 'dense_activ_4_2': hp.choice('dense_activ_4_2', ['relu', 'sigmoid']),
			 'dense_drop_4_2': hp.choice('dense_drop_4_2', np.arange(.2, .51, .05)),
			 'dense_units_4_3': hp.choice('dense_units_4_3', np.arange(25, 301, 25)),
			 'dense_activ_4_3': hp.choice('dense_activ_4_3', ['relu', 'sigmoid']),
			 'dense_drop_4_3': hp.choice('dense_drop_4_3', np.arange(.2, .51, .05)),
			 }
		]),
		#
		# training
		#
		'opt': hp.choice('opt', ['adam', 'sgd', 'rmsprop']),
		'opt_lr': hp.choice('opt_lr', [1e-05, 1e-04, 1e-03, 1e-02, 1e-01]),
		'batch_size': hp.choice('batch_size', np.arange(50, 210, 10)),
		#
		# seismic window
		#
		'win_size': hp.choice('win_size', np.arange(200, 501, 20)),
		'frac_dsamp_p1': hp.choice('frac_dsamp_p1', np.arange(.2, 1.1, .1)),
		'frac_dsamp_s1': hp.choice('frac_dsamp_s1', np.arange(.2, 1.1, .1)),
		'frac_dsamp_n1': hp.choice('frac_dsamp_n1', np.arange(.2, 1.1, .1)),
		#
		# filter / normalization on seismic waveform.
        # Integrates all the pre-processing tests into the optimization.
        # pre_mode = 1: no normalization + no filter
        # pre_mode = 2: no normalization + band-pass filter (2-10 Hz)
        # pre_mode = 3: no normalization + high-pass filter (freq_min = 0.2 Hz = 5 s)
		'pre_mode': hp.choice('pre_mode', [1, 2, 3]),
    }
    #
    return space


def get_space_pick_P():
    """
    returns hyperopt search space used in hyperparameter optimization of model trained for the P-phase picking task
    """
    #
    space = {
		#
		# recurrent layers
		#
		'rec_units_1': hp.choice('rec_units_1', np.arange(10, 210, 10)),
		'rec_drop_1_1': hp.choice('rec_drop_1_1', np.arange(.2, .51, .05)),
		'rec_drop_1_2': hp.choice('rec_drop_1_2', np.arange(.2, .51, .05)),
		'rec_nlay': hp.choice('rec_nlay', [
			{'layers': 'one'},
			{'layers': 'two',
             'rec_units_2': hp.choice('rec_units_2', np.arange(10, 210, 10)),
             'rec_drop_2_1': hp.choice('rec_drop_2_1', np.arange(.2, .51, .05)),
             'rec_drop_2_2': hp.choice('rec_drop_2_2', np.arange(.2, .51, .05)),
			 },
		]),
		#
		# general
		#
		# 'opt': hp.choice('opt', ['adam', 'sgd', 'rmsprop']),
		'opt_lr': hp.choice('opt_lr', [1e-05, 1e-04, 1e-03, 1e-02, 1e-01]),
		'batch_size': hp.choice('batch_size', np.arange(50, 210, 10)),
    }
    #
    return space


def get_space_pick_S():
    """
    returns hyperopt search space used in hyperparameter optimization of model trained for the S-phase picking task
    """
    #
    space = {
		#
		# recurrent layers
		#
		'rec_units_1': hp.choice('rec_units_1', np.arange(10, 210, 10)),
		'rec_drop_1_1': hp.choice('rec_drop_1_1', np.arange(.2, .51, .05)),
		'rec_drop_1_2': hp.choice('rec_drop_1_2', np.arange(.2, .51, .05)),
		'rec_nlay': hp.choice('rec_nlay', [
			{'layers': 'one'},
			{'layers': 'two',
             'rec_units_2': hp.choice('rec_units_2', np.arange(10, 210, 10)),
             'rec_drop_2_1': hp.choice('rec_drop_2_1', np.arange(.2, .51, .05)),
             'rec_drop_2_2': hp.choice('rec_drop_2_2', np.arange(.2, .51, .05)),
			 },
		]),
		#
		# general
		#
		# 'opt': hp.choice('opt', ['adam', 'sgd', 'rmsprop']),
		'opt_lr': hp.choice('opt_lr', [1e-05, 1e-04, 1e-03, 1e-02, 1e-01]),
		'batch_size': hp.choice('batch_size', np.arange(50, 210, 10)),
    }
    #
    return space


def make_prediction(model, st, dt, dct_trigger, dct_param):
    """
    retrieves P- and S-phase probabilities predicted by trained model.
    It generalizes function predict_GPD(), and should replace it eventually.
    """
    #
    # trim traces within common start and end times to avoid exception -->
    # ValueError: could not broadcast input array from shape (17958,460) into shape (17963,460)
    tstart_arr = np.array([tr.stats.starttime for tr in st])
    tend_arr = np.array([tr.stats.endtime for tr in st])
    tstart_cond = (tstart_arr == st[0].stats.starttime)
    tend_cond = (tend_arr == st[0].stats.endtime)
    st_trim_flag = False
    #
    if tstart_cond.sum() != len(tstart_arr) or tend_cond.sum() != len(tend_arr):
        print(f"strimming stream: {tstart_cond.sum()}, {tend_cond.sum()}")
        st_trim_flag = True
        st.trim(tstart_arr.max(), tend_arr.min())
    #
    #Reshaping data matrix for sliding window
    # print("Reshaping data matrix for sliding window")
    #
    st_data = [st[0].data, st[1].data, st[2].data]
    # print(st_data[0])
    if dct_param['st_normalized']:
        data_max = np.array([np.abs(tr.data).max() for tr in st]).max()
        for i, tr_data in enumerate(st_data):
            tr_data /= data_max
    # print(st_data[0])
    #
    tt = (np.arange(0, st_data[0].size, dct_trigger['n_shift']) + dct_trigger['n_win']) * dt #[sec]
    tt_i = np.arange(0, st_data[0].size, dct_trigger['n_shift']) + dct_trigger['n_feat'] #[samples??]
    #tr_win = np.zeros((tt.size, n_feat, 3))
    #
    try:
        sliding_N = sliding_window(st_data[0], dct_trigger['n_feat'], stepsize=dct_trigger['n_shift'])
        sliding_E = sliding_window(st_data[1], dct_trigger['n_feat'], stepsize=dct_trigger['n_shift'])
        sliding_Z = sliding_window(st_data[2], dct_trigger['n_feat'], stepsize=dct_trigger['n_shift'])
        tr_win = np.zeros((sliding_N.shape[0], dct_trigger['n_feat'], 3))
        tr_win[:,:,0] = sliding_N
        tr_win[:,:,1] = sliding_E
        tr_win[:,:,2] = sliding_Z
        #
        #normalization, separated into several operations to avoid memory errors
        aa = np.abs(tr_win)
        bb = np.max(aa, axis=(1,2))[:,None,None]
        tr_win = tr_win / bb
        #tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
        #
        tt = tt[:tr_win.shape[0]]
        tt_i = tt_i[:tr_win.shape[0]]
        #
        #make model predictions
        ts = model.predict(tr_win, verbose=True, batch_size=dct_trigger['batch_size'])
        #
        prob_P = ts[:,0]
        prob_S = ts[:,1]
        prob_N = ts[:,2]
    except ValueError:
        tt, ts, prob_S, prob_P, prob_N = [0],[0],[0],[0],[0]
    #     print(tt.shape, ts.shape, prob_S.shape, prob_P.shape, prob_N.shape)
    #
    return (tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st)


def calculate_trigger(ofile_path, st, net, sta, tt, ts, prob_P, prob_S, dct_trigger, best_params):
    """
    calculates trigger on and off times of P- and S- phases, from predicted P,S-class probability
    ...
    """
    #
    from obspy.signal.trigger import trigger_onset
    #trigger_onset(charfct, thres1, thres2, max_len=9e+99, max_len_delete=False)
    #calculates trigger on and off times from characteristic function charfct, given thresholds thres1 and thres2
    #
    #output file, containing predicted P- and S- triggered phases
    ofile = open(ofile_path, 'w')
    #
    #correction of time position for P, S predicted picks
    tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
    ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
    #
    #calculate trigger on and off times of P phases, from predicted P-class probability
    p_picks = []
    p_trigs = trigger_onset(prob_P, dct_trigger['pthres_p'][0], dct_trigger['pthres_p'][1], dct_trigger['max_trig_len'][0])
    # print("predicted picks...")
    for trig in p_trigs:
        if trig[1] == trig[0]:
            continue
        pick = np.argmax(ts[trig[0]:trig[1], 0])+trig[0]
        # stamp_pick = st[0].stats.starttime + tt[pick]
        stamp_pick = st[0].stats.starttime + tt[pick] + tp_shift
        # print(pick, tt[pick])
        # print(f"P {pick} {ts[pick][0]} {stamp_pick} {tp_shift:.2f}")
        p_picks.append((stamp_pick, pick))
        # ofile.write("%s %s P %s\n" % (net, sta, stamp_pick.isoformat()))
        ofile.write("%s %s P %s %s\n" % (net, sta, stamp_pick.isoformat(), f"{ts[pick][0]}"))
    #
    #calculate trigger on and off times of S phases, from predicted S-class probability
    s_picks = []
    s_trigs = trigger_onset(prob_S, dct_trigger['pthres_s'][0], dct_trigger['pthres_s'][1], dct_trigger['max_trig_len'][1])
    # print(s_trigs)
    for trig in s_trigs:
        if trig[1] == trig[0]:
            continue
        pick = np.argmax(ts[trig[0]:trig[1], 1])+trig[0]
        # stamp_pick = st[0].stats.starttime + tt[pick]
        stamp_pick = st[0].stats.starttime + tt[pick] + ts_shift
        # print(f"S {pick} {ts[pick][1]} {stamp_pick} {ts_shift:.2f}")
        s_picks.append((stamp_pick, pick))
        # ofile.write("%s %s S %s\n" % (net, sta, stamp_pick.isoformat()))
        ofile.write("%s %s S %s %s\n" % (net, sta, stamp_pick.isoformat(), f"{ts[pick][1]}"))
    #
    ofile.close()
    return p_picks, s_picks, p_trigs, s_trigs


def run_prediction(best_model, best_params, ts, dct_sta, dct_param, dct_trigger, opath_tmp, flag_data):
    """
    retrieve continuous waveforms and run prediction on them
    """
    #
    tstart, tend, dt_iter, dt_win = ts
    #
    t_iters = []
    tstart_iter = tstart
    tend_iter = tstart_iter + dt_iter
    #
    print("parameters for waveform processing:")
    print([f"{k}: {dct_param[k]}" for k in dct_param.keys() if k not in ['start_off','stop_off','dt_search']])
    #
    print("#")
    while tstart_iter < tend:
        t_iters.append([tstart_iter, tend_iter])
        tstart_iter += dt_iter
        tend_iter += dt_iter
    #
    print(f"preparing time windows ({len(t_iters)}) for iteration over continuous waveforms...")
    for t in t_iters:
        print(t)
    print("")
    #
    dct_st = {}
    stas = dct_sta['stas']
    for i, t_iter in enumerate(t_iters):
        #
        tstart_iter, tend_iter = t_iter
        # print(t_iter)
        # print(tstart_iter, tend_iter)
        sts = [oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream()]
        st = oc.Stream()
        #
        twin_str = f"{tstart_iter.year}{tstart_iter.month:02}{tstart_iter.day:02}"
        twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}"
        twin_str += f"_{(tend_iter-1.).hour:02}{(tend_iter-1.).minute:02}"
        opath = f"{opath_tmp}/{flag_data}/wf_{twin_str}"
        yy = tstart_iter.year
        doy = '%03d' % (tstart_iter.julday)
        #
        print("")
        print("retrieving seismic waveforms for stations:")
        print(stas)
        #
        st_arg = 0
        stas_remove = []
        for j, sta in enumerate(stas):
            #
            # retrieve waveforms
            #
            net = dct_sta['net']
            ch = dct_sta['ch']
            flist = glob('archive/'+str(yy)+'/'+net+'/'+sta+'/'+ch+'?.D/*.'+doy)
            #
            if len(flist) > 0:
                outstr = f"seismic data found for: net = {net}, sta = {sta}, st_count = {len(sts[st_arg])}, st_arg = {st_arg}"
                print(outstr)
                outstr = str(flist)
                print(outstr)
            else:
                outstr = f"seismic data not found for: net = {net}, sta = {sta}"
                print(outstr)
            #
            if len(sts[st_arg]) >= 50:
                st_arg += 1
            #
            for f in flist:
                if sta == 'LVC' and f.split(sta)[-1][1:3] == '00':
                    continue
                else:
                    pass
                #
                # print(f)
                #
                try:
                    # st += oc.read(f)
                    tr = oc.read(f)
                    #
                    # using only traces with sampling_rate = 100 hz
                    # TODO: adapt ??
                    if tr[0].stats.sampling_rate != 100.:
                        print(f"skipping trace:")
                        print(tr[0])
                        continue
                    #
                    if len(tr) > 1:  # merge duplicate traces
                        try:
                            tr.merge()
                        except:
                            # catch Exception: Can't merge traces with same ids but differing sampling rates!
                            #
                            outstr = f"Exception: traces with same ids but differing sampling rates!"
                            print(outstr)
                            # sr_mean = float(int(np.array([trr.stats.sampling_rate for trr in tr]).mean()))
                            sr_mean = round(np.array([trr.stats.sampling_rate for trr in tr]).mean(), ndigits=1)
                            for trr in tr:
                                dsr = abs(trr.stats.sampling_rate - sr_mean)
                                if dsr < 1.:
                                    # trick much faster than resampling for almost identical sampling rates
                                    trr.stats.sampling_rate = float(int(tr[0].stats.sampling_rate))
                                    outstr = f"{trr} --> setting sampling rate to {sr_mean} (dsr = {dsr})"
                                    print(outstr)
                                else:
                                    outstr = f"{trr} --> resampling to {sr_mean}"
                                    print(outstr)
                                    trr.resample(sr_mean)
                            try:
                                tr.merge()
                            except:
                                # catch Exception: Can't merge traces with same ids but differing data types!
                                outstr = f"Exception: can't merge traces with same ids but differing data types!"
                                print(outstr)
                                for trr in tr:
                                    trr.data = trr.data.astype(np.int32)
                                tr.merge()
                    sts[st_arg] += tr
                #
                except InternalMSEEDError:
                    stas_remove.append(sta)
                    outstr = f"skipping {f} --> InternalMSEEDError exception"
                    print(outstr)
                    continue
        #
        for stt in sts:
            st += stt
        del sts
        #
        stas_remove = set(stas_remove)
        # print(stas_remove)
        for s in stas_remove:
            for tr in st.select(station=s):
                st.remove(tr)
        # print(len(st))
        print(st.__str__(extended=True))
        #
        # process raw stream data
        stt = st.copy()
        del st
        # print(stt[0].stats.sampling_rate)
        #
        print("#")
        print("processing raw stream data...")
        print('detrend...')
        if dct_param['st_detrend']:
            try:
                stt.detrend(type='linear')
            except NotImplementedError:
                # catch exception NotImplementedError: Trace with masked values found. This is not supported for this operation.
                # Try the split() method on Trace/Stream to produce a Stream with unmasked Traces.
                stt = stt.split()
                stt.detrend(type='linear')
            except ValueError:
                # catch exception ValueError: array must not contain infs or NaNs.
                # Due to presence of e.g. nans in at least one trace data.
                for tr in stt:
                    nnan = np.count_nonzero(np.isnan(tr.data))
                    ninf = np.count_nonzero(np.isinf(tr.data))
                    if nnan > 0:
                        print(f"{tr} --> removed (due to presence of nans)")
                        stt.remove(tr)
                        continue
                    if ninf > 0:
                        print(f"{tr} --> removed (due to presence of infs)")
                        stt.remove(tr)
                        continue
        #
        # stt.detrend(type='linear')
        print('filter...')
        if dct_param['st_filter']:
            if dct_param['filter'] == 'bandpass':
                stt.filter(type=dct_param['filter'], freqmin=dct_param['freq_min'], freqmax=dct_param['freq_max'])
            elif dct_param['filter'] == 'highpass':
                stt.filter(type=dct_param['filter'], freq=dct_param['freq_min'])
        # print('resample')
        # stt.resample(dct_param['freq_resample'])
        # stt.sort(['location','channel'])
        #
        print('merging...')
        stt.merge()
        #
        dct_st[i+1] = {}
        tstart_win = tstart_iter
        tend_win = tstart_iter + dt_win
        nwin = 1
        #
        print("#")
        print("extracting time windows from waveforms...")
        while tstart_win < tend_iter:
            #
            # sttt = stt.copy()
            # sttt.trim(tstart_win, tend_win)
            sttt = stt.slice(tstart_win, tend_win)
            dct_st[i+1][nwin] = {
                'stt': {}, 'pred': {},
                'twin': [tstart_win, tend_win],
            }
            #
            for t, tr in enumerate(sttt):
                sta = tr.stats.station
                if sta not in dct_st[i+1][nwin]['stt'].keys():
                    # dct_st[i+1][nwin]['st'][sta] = oc.Stream()
                    # dct_st[i+1][nwin]['st'][sta] += tr
                    #
                    dct_st[i+1][nwin]['stt'][sta] = oc.Stream()
                    dct_st[i+1][nwin]['stt'][sta] += tr
                else:
                    # dct_st[i+1][nwin]['st'][sta] += tr
                    dct_st[i+1][nwin]['stt'][sta] += tr
            #
            dct_st[i+1][nwin]['twin'] = [tstart_win, tend_win]
            print(nwin, tstart_win, tend_win)
            tstart_win += dt_win
            tend_win += dt_win
            nwin += 1
        #
        # predict seismic phases on processed stream data
        for k in dct_st[i+1].keys():
            for s in sorted(dct_st[i+1][k]['stt'].keys())[:]:
                st_tmp = dct_st[i+1][k]['stt'][s]
                net = st_tmp[0].stats.network
                ch = st_tmp[0].stats.channel
                dt = st_tmp[0].stats.delta
                print("#")
                print(f"Calculating predictions for stream in window {k:03}/{len(list(dct_st[i+1].keys())):03}: {net}.{s}..{ch[:-1]}?...")
                print(st_tmp)
                #
                # predict
                # tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = make_prediction(best_model, st_tmp, dt, dct_trigger, dct_param)
                tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = make_prediction(best_model, st_tmp, dt, dct_trigger, dct_param)
                # print(tt.shape, ts.shape, prob_S.shape, prob_P.shape, prob_N.shape)
                #
                # skip streams raising ValueError in make_prediction()
                if len(ts) == 1 and ts[0] == 0:
                    val_err = f"Sliding window size may not exceed size of selected axis"
                    print(f"skipping stream {net}.{s}..{ch[:-1]}? due to ValueError: {val_err}...")
                    continue
                #
                # trigger picks
                opath2 = f"{opath}/{k:03}"
                os.makedirs(f"{opath2}/pick_stats", exist_ok=True)
                ofile_path = f"{opath2}/pick_stats/{s}_pick_detected"
                p_picks, s_picks, p_trigs, s_trigs = calculate_trigger(ofile_path, st_tmp, net, s, tt, ts, prob_P, prob_S, dct_trigger, best_params)
                print(f"p_picks = {len(p_picks)}, s_picks = {len(s_picks)}")
                dct_st[i+1][k]['pred'][s] = {
                    'dt': dt, 'tt': tt, 'ts': ts,
                    'p_picks': p_picks, 's_picks': s_picks,
                    'p_trigs': p_trigs, 's_trigs': s_trigs,
                    'opath': opath2,
                }
                if st_trim_flag:
                    dct_st[i+1][k]['stt'][s] = st_trimmed
    #
    return dct_st


def get_picks_plot(dct_st, dct_trigger, tp_shift, ts_shift, flag_data):
    """
    retrieve associated P, S predicted (ML) and detected (STA/LTA) picks
    """
    from shapely import geometry as geom
    #
    dt_PS_max = dct_trigger['detec_thres'][flag_data]['dt_PS_max'] # seconds
    #
    p_picks = [pick[0] for pick in dct_st['p_picks']]
    s_picks = [pick[0] for pick in dct_st['s_picks']]
    p_trigs = np.array([trig for trig in dct_st['p_trigs'] if trig[1] != trig[0]])
    s_trigs = np.array([trig for trig in dct_st['s_trigs'] if trig[1] != trig[0]])
    print(f"detected picks (P, S): {len(p_picks)}, {len(p_trigs)}, {len(s_picks)}, {len(s_trigs)}")
    tt_p_arg = [pick[1] for pick in dct_st['p_picks']]
    tt_s_arg = [pick[1] for pick in dct_st['s_picks']]
    tt = dct_st['tt']
    prob_P = dct_st['ts'][:,0]
    prob_S = dct_st['ts'][:,1]
    prob_N = dct_st['ts'][:,2]
    #
    tpicks_ml_p = np.array([t.timestamp for t in p_picks])
    tpicks_ml_s = np.array([t.timestamp for t in s_picks])
    #
    # print(p_picks)
    # print(sorted(p_picks))
    # print(s_picks)
    # print(sorted(s_picks))
    # print(tpicks_ml_p)
    # print(tpicks_ml_s)
    #
    p_picks_bool = np.full(len(tpicks_ml_p), True)
    s_picks_bool = np.full(len(tpicks_ml_s), True)
    s_arg_used = []
    #
    # (1) iterate over predicted P picks, which are nearby S picks
    #
    for i, tp in enumerate(tpicks_ml_p[:]):
        #
        # search S picks detected nearby P phases
        #
        cond_pre = prob_P[:tt_p_arg[i]] > .5
        cond_pre = cond_pre[::-1]
        # arg_th_pre = tt_p_arg[i] - np.argmin(cond_pre)
        tp_th_pre = tp - (np.argmin(cond_pre) * dct_trigger['n_shift'] * dct_trigger['only_dt']) - dct_trigger['detec_thres'][flag_data]['tp_th_add']
        #
        cond_pos = prob_P[tt_p_arg[i]:] > .5
        # arg_th_pos = tt_p_arg[i] + np.argmin(cond_pos)
        tp_th_pos = tp + (np.argmin(cond_pos) * dct_trigger['n_shift'] * dct_trigger['only_dt']) + dct_trigger['detec_thres'][flag_data]['tp_th_add']
        #
        ts_in_th = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss >= tp_th_pre and tss <= tp_th_pos]
        #
        # picks detected before and after current P pick
        #
        tp_in_next = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp >= tp_th_pos and tpp <= tp + dt_PS_max]
        ts_in_next = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss >= tp_th_pos and tss <= tp + dt_PS_max]
        tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > tp - dt_PS_max and tpp < tp_th_pre]
        # ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > tp - dt_PS_max*.8 and tss < tp_th_pre]
        #
        if len(ts_in_th) > 0:
            # pick = P/S or S/P
            #
            s_arg_used.append(ts_in_th[0][0])
            #
            if prob_P[tt_p_arg[i]] >= prob_S[tt_s_arg[ts_in_th[0][0]]]:
                # pick --> P
                #
                s_picks_bool[ts_in_th[0][0]] = False
            else:
                # pick --> S ??
                #
                if (len(tp_in_next) > 0 or len(ts_in_next) > 0):
                    #
                    if len(tp_in_prior) == 0:
                        #
                        # pick_next found, pick_prior not found; pick --> P
                        s_picks_bool[ts_in_th[0][0]] = False
                    #
                    if len(tp_in_prior) > 0:
                        p_picks_bool_prior = [tpp[0] for t, tpp in enumerate(tp_in_prior) if p_picks_bool[tpp[0]]]
                        if len(p_picks_bool_prior) == 0:
                            #
                            # pick_next found, pick_prior not found; pick --> P
                            s_picks_bool[ts_in_th[0][0]] = False
                        #
                        else:
                            # pick_next found, pick_prior found; pick --> S
                            p_picks_bool[i] = False
                #
                else:
                    if len(tp_in_prior) == 0:
                        # pick_next not found, pick_prior not found;
                        # pick --> possibly actual S, but P not detected --> discard P and S
                        p_picks_bool[i] = False
                        s_picks_bool[ts_in_th[0][0]] = False
                    #
                    else:
                        p_picks_bool_prior = [tpp[0] for t, tpp in enumerate(tp_in_prior) if p_picks_bool[tpp[0]]]
                        if len(p_picks_bool_prior) == 0:
                            #
                            # pick_next not found, pick_prior not found; pick --> discard P and S
                            p_picks_bool[i] = False
                            s_picks_bool[ts_in_th[0][0]] = False
                        #
                        else:
                            # pick_next not found, pick_prior found; pick --> S
                            p_picks_bool[i] = False
    #
    # (2) iterate over predicted S picks, which were not handled in iteration over P picks done in (1)
    # --> S picks for which there is no earlier P or P-S picks selected will be discarded
    #
    s_arg_nonused = [i for i, ts in enumerate(tpicks_ml_s) if i not in s_arg_used]
    for i in s_arg_nonused:
        #
        ts = tpicks_ml_s[i]
        #
        # P picks detected before current S pick
        #
        tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > ts - dt_PS_max and tpp < ts and p_picks_bool[t]]
        # ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > ts - dt_PS_max*.8 and tss < ts and s_picks_bool[t]]
        #
        if len(tp_in_prior) == 0:
            #
            # prior pick not found --> discard
            s_picks_bool[i] = False
        #
        if len(tp_in_prior) > 0:
            #
            tp_prior = tp_in_prior[-1][1]
            # ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > tp and tss < ts and t in s_arg_selected]
            ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > tp_prior and tss < ts and t in np.where(s_picks_bool)[0]]
            #
            if len(ts_in_prior) > 1:
                s_picks_bool[i] = False
            #
            if len(ts_in_prior) == 1:
                #
                ts_prior = ts_in_prior[0][1]
                if ts > ts_prior + abs(tp_prior - ts_prior):
                    s_picks_bool[i] = False
    #
    # 3) iterate over selected S picks in order to resolve between possible duplicated S phases
    # --> this is special case SC1...
    #
    s_arg_selected = np.where(s_picks_bool)[0]
    dt_max_s_dup = dct_trigger['detec_thres'][flag_data]['dt_max_s_dup'] # seconds
    s_arg_used_dup = []
    dct_s_dup = {}
    #
    for i, s_arg in enumerate(s_arg_selected):
        #
        dct_s_dup[s_arg] = [s_arg]
        ts = tpicks_ml_s[s_arg]
        cond_pos = prob_S[tt_s_arg[s_arg]:] > .5
        ts_th_pos = ts + (np.argmin(cond_pos) * dct_trigger['n_shift'] * dct_trigger['only_dt'])
        #
        for j, s_arg2 in enumerate(s_arg_selected[i+1: len(s_arg_selected)]):
            #
            ts2 = tpicks_ml_s[s_arg2]
            cond_pre = prob_S[:tt_s_arg[s_arg2]] > .5
            cond_pre = cond_pre[::-1]
            ts2_th_pre = ts2 - (np.argmin(cond_pre) * dct_trigger['n_shift'] * dct_trigger['only_dt'])
            #
            if abs(ts_th_pos - ts2_th_pre) < dt_max_s_dup:
                dct_s_dup[s_arg].append(s_arg2)
            else:
                break
    #
    # for possible duplicated S phases, unselect presumed false ones
    for s_arg in dct_s_dup:
        if len(dct_s_dup[s_arg]) > 1:
            s_dup_pb = np.array([prob_S[tt_s_arg[s_arg_dup]] for s_arg_dup in dct_s_dup[s_arg]])
            s_dup_pb_argmax = np.argmax(s_dup_pb)
            s_arg_false = [s_arg3 for s_arg3 in dct_s_dup[s_arg] if s_arg3 != dct_s_dup[s_arg][s_dup_pb_argmax]]
            for s_false in s_arg_false:
                s_picks_bool[s_false] = False
                s_arg_used_dup.append(s_false)
    #
    p_arg_selected = np.where(p_picks_bool)[0]
    s_arg_selected = np.where(s_picks_bool)[0]
    #
    # 4) iterate over non-selected S picks in order to resolve possible missing S picks from event with double S phase
    # --> this is special case SC3...
    # --> this could be included in criterion 2)
    #
    s_arg_nonselected = np.where(~s_picks_bool)[0]
    #
    for i, s_arg in enumerate(s_arg_nonselected):
        #
        if s_arg in s_arg_used_dup:
            continue
        #
        ts = tpicks_ml_s[s_arg]
        tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > ts - dt_PS_max*1.5 and tpp < ts and t in p_arg_selected]
        #
        if len(tp_in_prior) == 1:
            #
            tp_prior = tp_in_prior[0][1]
            # ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > tp and tss < ts and t in s_arg_selected]
            ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > tp_prior and tss < ts and t in np.where(s_picks_bool)[0]]
            if len(ts_in_prior) == 1:
                #
                ts_prior = ts_in_prior[0][1]
                if ts <= ts_prior + abs(tp_prior - ts_prior):
                    #
                    # possibly event with double S phase if predicted prior P and S picks have been selected at t, where:
                    # 1) ts - dt_PS_max*1.5 < t < ts
                    # 2) ts <= ts_prior + abs(tp_prior - ts_prior);
                    # then pick --> S
                    s_picks_bool[s_arg] = True
    #
    s_arg_selected = np.where(s_picks_bool)[0]
    s_arg_nonselected = np.where(~s_picks_bool)[0]
    #
    # 5) iterate over selected S picks in order to resolve between nearby P/S
    # phases which were not found before in (1)
    # --> this is special case SC2b
    #
    dt_max_sp_near = dct_trigger['detec_thres'][flag_data]['dt_max_sp_near'] # seconds
    dct_sp_near = {}
    #
    for i, s_arg in enumerate(s_arg_selected):
        #
        dct_sp_near[s_arg] = []
        ts = tpicks_ml_s[s_arg]
        #
        s_cond_pos = prob_S[tt_s_arg[s_arg]:] > .5
        ts_th_pos = ts + (np.argmin(s_cond_pos) * dct_trigger['n_shift'] * dct_trigger['only_dt'])
        #
        s_cond_pre = prob_S[:tt_s_arg[s_arg]] > .5
        s_cond_pre = s_cond_pre[::-1]
        ts_th_pre = ts - (np.argmin(s_cond_pre) * dct_trigger['n_shift'] * dct_trigger['only_dt'])
        #
        for j, p_arg in enumerate(p_arg_selected):
            #
            tp = tpicks_ml_p[p_arg]
            #
            p_cond_pos = prob_P[tt_p_arg[p_arg]:] > .5
            tp_th_pos = tp + (np.argmin(p_cond_pos) * dct_trigger['n_shift'] * dct_trigger['only_dt'])
            #
            p_cond_pre = prob_P[:tt_p_arg[p_arg]] > .5
            p_cond_pre = p_cond_pre[::-1]
            tp_th_pre = tp - (np.argmin(p_cond_pre) * dct_trigger['n_shift'] * dct_trigger['only_dt'])
            #
            dt_s_then_p = abs(ts_th_pos - tp_th_pre)
            dt_p_then_s = abs(tp_th_pos - ts_th_pre)
            #
            if dt_s_then_p < dt_max_sp_near or dt_p_then_s < dt_max_sp_near:
                dct_sp_near[s_arg].append([p_arg, min(dt_s_then_p, dt_p_then_s)])
    #
    # for possible nearby P/S phases, unselect presumed false ones
    for s_arg in dct_sp_near:
        if len(dct_sp_near[s_arg]) > 0:
            #
            s_near_pb = prob_S[tt_s_arg[s_arg]]
            p_near_pb_arg = np.argmin([p_near[1] for p_near in dct_sp_near[s_arg]])
            p_near_arg = dct_sp_near[s_arg][p_near_pb_arg][0]
            p_near_pb = prob_P[tt_p_arg[p_near_arg]]
            #
            if s_near_pb >= p_near_pb:
                p_picks_bool[p_near_arg] = False
            else:
                s_picks_bool[s_arg] = False
    #
    p_arg_selected = np.where(p_picks_bool)[0]
    s_arg_selected = np.where(s_picks_bool)[0]
    #
    print(f"selected picks (P, S): {len(np.where(p_picks_bool)[0])}, {len(np.where(s_picks_bool)[0])}")
    print("#")
    for i, tp in enumerate(tpicks_ml_p):
        print(i+1, tp, p_picks_bool[i])
    #
    print("#")
    for i, ts in enumerate(tpicks_ml_s):
        print(i+1, ts, s_picks_bool[i])
    #
    # fill dictionary with selected picks
    #
    pick_asoc = {
        'P': {}, 'S': {},
    }
    pick_asoc['P']['pick'] = tpicks_ml_p
    pick_asoc['P']['trig'] = p_trigs
    pick_asoc['P']['pb'] = np.array([prob_P[tt_arg] for i, tt_arg in enumerate(tt_p_arg)])
    pick_asoc['P']['bool'] = p_picks_bool
    pick_asoc['P']['true_arg'] = p_arg_selected
    #
    pick_asoc['S']['pick'] = tpicks_ml_s
    pick_asoc['S']['trig'] = s_trigs
    pick_asoc['S']['pb'] = np.array([prob_S[tt_arg] for i, tt_arg in enumerate(tt_s_arg)])
    pick_asoc['S']['bool'] = s_picks_bool
    pick_asoc['S']['true_arg'] = s_arg_selected
    #
    # estimate area under S-phase probability series of selected picks
    #
    pb_thres = dct_trigger['detec_thres'][flag_data]['pb_thres']
    pick_asoc['S']['pb_S_arg'] = []
    pick_asoc['S']['pb_shape'] = []
    pick_asoc['S']['pb_area'] = []
    pick_asoc['S']['pb'] = []
    #
    for ts_arg in pick_asoc['S']['true_arg']:
        #
        pb_arg_1 = tt_s_arg[ts_arg]
        polygon_1 = []
        while (prob_S[pb_arg_1] >= .5):
            pb_arg_1 -= 1
            if pb_arg_1 >= 0:
                polygon_1.append((tt[pb_arg_1]+ts_shift, prob_S[pb_arg_1]))
            else:
                polygon_1.append((tt[pb_arg_1+1]+ts_shift, pb_thres-.1))
                break
        #
        pb_arg_2 = tt_s_arg[ts_arg]
        polygon_2 = []
        flag_index_err = False
        while (prob_S[pb_arg_2] >= .5):
            pb_arg_2 += 1
            try:
                polygon_2.append((tt[pb_arg_2]+ts_shift, prob_S[pb_arg_2]))
            except IndexError:
                pb_arg_2 -= 1
                polygon_2.append((tt[pb_arg_2]+ts_shift, pb_thres-.1))
                break
        #
        pb_arg_0 = tt_s_arg[ts_arg]
        polygon_0 = [(tt[pb_arg_0]+ts_shift, prob_S[pb_arg_0])]
        polygon = polygon_1[::-1] + polygon_0 + polygon_2
        shape_polygon = geom.Polygon(polygon)
        if pb_arg_1 < 0:
            pb_arg_1 = 1
        #
        try:
            pb_thres_line = [(tt[pb_arg_1-1]+ts_shift, pb_thres), (tt[pb_arg_2+1]+ts_shift, pb_thres)]
        except IndexError:
            pb_thres_line = [(tt[pb_arg_1-1]+ts_shift, pb_thres), (tt[pb_arg_2]+ts_shift, pb_thres)]
        shape_line = geom.LineString(pb_thres_line)
        # print(tt[tt_s_arg[ts_arg-1]], tt[tt_s_arg[ts_arg]], tt[tt_s_arg[ts_arg+1]])
        # print(tt)
        # print(polygon_1)
        # print(polygon)
        shape_intersec = shape_polygon.intersection(shape_line)
        vertexes = shape_intersec.coords[:]
        # print(type(vertexes))
        # print(type(vertexes[:]))
        # print(vertexes[:])
        polygon_pb = [vertexes[0]] + polygon_1[::-1][1:] + polygon_0 + polygon_2[:-1] + [vertexes[1]]
        shape_polygon_pb = geom.Polygon(polygon_pb)
        #
        pick_asoc['S']['pb'].append(prob_S[tt_s_arg[ts_arg]])
        pick_asoc['S']['pb_area'].append(shape_polygon_pb.area)
        pick_asoc['S']['pb_shape'].append({
            'polygon': shape_polygon, 'intersec': shape_intersec,
            'polygon_pb': shape_polygon_pb,
        })
        #
        pick_asoc['S']['pb_S_arg'].append((pb_arg_1, pb_arg_2))
    #
    return pick_asoc


def plot_predicted_phase_P(dct_pick, dct_mc, data, sta, tpick_det, opath, plot_num, save_plot=True):
    """
    plot predicted P phase and pick onset
    """
    #
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    #
    opath_fig = f"{opath}/pick_plots"
    os.makedirs(opath_fig, exist_ok=True)
    #
    tpick_pred = dct_pick['tpick']
    tpick_pred_th1 = dct_pick['tpick_th1']
    tpick_pred_th2 = dct_pick['tpick_th2']
    terr_pre = dct_pick['terr_pre']
    terr_pos = dct_pick['terr_pos']
    pick_class = dct_pick['pick_class']
    mc_pred = dct_mc['mc_pred']
    mc_pred_mean = dct_mc['mc_pred_mean']
    mc_pred_mean_arg_pick = dct_mc['mc_pred_mean_arg_pick']
    mc_pred_std_pick = dct_mc['mc_pred_std_pick']
    prob_th1 = dct_mc['prob_th1']
    prob_th2 = dct_mc['prob_th2']
    #
    # plot - phase window input for network
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    samp_freq = 100.
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / samp_freq
    #
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=1.)
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='r', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='r', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"comp Z"
    # ax[-1].text(0.02, .95, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = 0.
    xmax = tr_win_x.max()
    # title_str = f"time residual (tpick_true - tpick_pred) = {tres:.3f}"
    # ax[-1].set_title(title_str)
    ax[-1].set_xlim([xmin, xmax])
    ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .5))
    ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase P: {opath_fig}/{sta}_pred_P_{plot_num+1:02}.png")
    print("#")
    ofig = f"{opath_fig}/{sta}_pred_P_{plot_num+1:02}"
    # plt.savefig(f"{ofig}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (zoom around predicted time pick)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2., zorder=1)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    # ax_tmp.plot(tr_win_x, best_pred_prob[i,:,0], c='magenta', lw=1.)
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--', zorder=1)
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1., zorder=1)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
    ax_tmp.yaxis.set_minor_locator(ticker.MultipleLocator(.05))
    ax_tmp.axhline(mc_pred_mean[mc_pred_mean_arg_pick,0], c='magenta', lw=1., ls='--', zorder=2)
    ax_tmp.axhline(prob_th1, c='magenta', lw=1., ls='--', zorder=2)
    ax_tmp.axhline(prob_th2, c='magenta', lw=1., ls='--', zorder=2)
    #
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='r', lw=1.5, ls='-', clip_on=False, zorder=3)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='r', lw=1.5, ls='--', clip_on=False, zorder=3)
    ax[-1].vlines(x=tpick_pred_th1, ymin=-1., ymax=1., color='r', lw=1.5, ls=':', clip_on=False, zorder=3)
    ax[-1].vlines(x=tpick_pred_th2, ymin=-1., ymax=1., color='r', lw=1.5, ls=':', clip_on=False, zorder=3)
    # ax[-1].vlines(x=tpick_pred-tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].vlines(x=tpick_pred+tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # arg_pred = mc_pred_mean_arg_pick
    # tr_label_1 = f"tpick_true = {tpick_true:.3f}"
    tr_label_1 = f"tpred = {tpick_pred:.3f}"
    tr_label_2 = f"terr(1 x pb_std) = (-{terr_pre:.3f}, +{terr_pos:.3f})"
    tr_label_3 = f"pick_class = {pick_class}"
    tr_label_4 = f"pb, pb_std = ({mc_pred_mean[mc_pred_mean_arg_pick,0]:.3f}, {mc_pred_std_pick:.3f})"
    # ax[-1].text(0.01, .975, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .935, tr_label_2, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .895, tr_label_3, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .855, tr_label_4, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = tpick_pred - .5
    xmax = tpick_pred + .5
    ax[-1].set_xlim([xmin, xmax])
    tick_major = np.arange(xmin, xmax + .1, .1)
    tick_minor = np.arange(xmin, xmax + .01, .02)
    # ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .1))
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    # ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.05))
    # ax[-1].xaxis.set_minor_locator(ticker.LinearLocator(5))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase P: {opath_fig}/{sta}_pred_P_mc_{plot_num+1:02}.png")
    print(tr_label_1)
    print(tr_label_2)
    print(tr_label_3)
    print(tr_label_4)
    print("#")
    ofig = f"{opath_fig}/{sta}_pred_P_mc_{plot_num+1:02}"
    # plt.savefig(f"{ofig}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()


def plot_predicted_phase_S(dct_pick, dct_mc, data, sta, tpick_det, opath, plot_num, save_plot=True):
    """
    plot predicted S phase and pick onset
    """
    #
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    #
    opath_fig = f"{opath}/pick_plots"
    os.makedirs(opath_fig, exist_ok=True)
    #
    tpick_pred = dct_pick['tpick']
    tpick_pred_th1 = dct_pick['tpick_th1']
    tpick_pred_th2 = dct_pick['tpick_th2']
    terr_pre = dct_pick['terr_pre']
    terr_pos = dct_pick['terr_pos']
    pick_class = dct_pick['pick_class']
    mc_pred = dct_mc['mc_pred']
    mc_pred_mean = dct_mc['mc_pred_mean']
    mc_pred_mean_arg_pick = dct_mc['mc_pred_mean_arg_pick']
    mc_pred_std_pick = dct_mc['mc_pred_std_pick']
    prob_th1 = dct_mc['prob_th1']
    prob_th2 = dct_mc['prob_th2']
    #
    # plot - phase window input for network (comp E)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    samp_freq = 100.
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / samp_freq
    print(tr_win_x.shape, tr_win_y.shape)
    #
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=1.)
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"comp E"
    # ax[-1].text(0.02, .95, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = 0.
    xmax = tr_win_x.max()
    # title_str = f"time residual (tpick_true - tpick_pred) = {tres:.3f}"
    # ax[-1].set_title(title_str)
    ax[-1].set_xlim([xmin, xmax])
    ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .5))
    ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_E_{plot_num+1:02}.png")
    print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_E_{plot_num+1:02}"
    # plt.savefig(f"{ofig}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (comp N)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,1]
    tr_win_x = np.arange(tr_win_y.shape[0]) / samp_freq
    print(tr_win_x.shape, tr_win_y.shape)
    #
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=1.)
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"comp N"
    # ax[-1].text(0.02, .95, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = 0.
    xmax = tr_win_x.max()
    # title_str = f"time residual (tpick_true - tpick_pred) = {tres:.3f}"
    # ax[-1].set_title(title_str)
    ax[-1].set_xlim([xmin, xmax])
    ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .5))
    ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_N_{plot_num+1:02}.png")
    print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_N_{plot_num+1:02}"
    # plt.savefig(f"{ofig}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (zoom around predicted time pick, comp E)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / samp_freq
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2.)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    # ax_tmp.plot(tr_win_x, best_pred_prob[i,:,0], c='magenta', lw=1.)
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--')
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1.)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
    ax_tmp.yaxis.set_minor_locator(ticker.MultipleLocator(.05))
    ax_tmp.axhline(mc_pred_mean[mc_pred_mean_arg_pick,0], c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th1, c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th2, c='magenta', lw=1., ls='--')
    #
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th1, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th2, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    # ax[-1].vlines(x=tpick_pred-tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].vlines(x=tpick_pred+tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # arg_pred = mc_pred_mean_arg_pick
    # tr_label_1 = f"tpick_true = {tpick_true:.3f}"
    tr_label_1 = f"tpred = {tpick_pred:.3f}"
    tr_label_2 = f"terr(1 x pb_std) = (-{terr_pre:.3f}, +{terr_pos:.3f})"
    tr_label_3 = f"pick_class = {pick_class}"
    tr_label_4 = f"pb, pb_std = ({mc_pred_mean[mc_pred_mean_arg_pick,0]:.3f}, {mc_pred_std_pick:.3f})"
    # ax[-1].text(0.01, .975, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .935, tr_label_2, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .895, tr_label_3, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .855, tr_label_4, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = tpick_pred - .5
    xmax = tpick_pred + .5
    ax[-1].set_xlim([xmin, xmax])
    tick_major = np.arange(xmin, xmax + .1, .1)
    tick_minor = np.arange(xmin, xmax + .01, .02)
    # ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .1))
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    # ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.05))
    # ax[-1].xaxis.set_minor_locator(ticker.LinearLocator(5))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_E_mc_{plot_num+1:02}.png")
    print(tr_label_1)
    print(tr_label_2)
    print(tr_label_3)
    print(tr_label_4)
    print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_E_mc_{plot_num+1:02}"
    # plt.savefig(f"{ofig}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (zoom around predicted time pick, comp N)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,1]
    tr_win_x = np.arange(tr_win_y.shape[0]) / samp_freq
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2.)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    # ax_tmp.plot(tr_win_x, best_pred_prob[i,:,0], c='magenta', lw=1.)
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--')
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1.)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
    ax_tmp.yaxis.set_minor_locator(ticker.MultipleLocator(.05))
    ax_tmp.axhline(mc_pred_mean[mc_pred_mean_arg_pick,0], c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th1, c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th2, c='magenta', lw=1., ls='--')
    #
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th1, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th2, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    # ax[-1].vlines(x=tpick_pred-tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].vlines(x=tpick_pred+tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"tpick_pred = {tpick_pred:.3f}"
    # tr_label_2 = f"terr(1 x prob_std) = (-{terr_pre:.3f}, +{terr_pos:.3f})"
    # tr_label_3 = f"pick_class = {pick_class}"
    # tr_label_4 = f"prob, prob_std = ({mc_pred_mean[mc_pred_mean_arg_pick,0]:.3f}, {mc_pred_std_pick:.3f})"
    # ax[-1].text(0.02, .975, tr_label_1, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.02, .935, tr_label_2, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.02, .895, tr_label_3, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.02, .855, tr_label_4, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = tpick_pred - .5
    xmax = tpick_pred + .5
    ax[-1].set_xlim([xmin, xmax])
    tick_major = np.arange(xmin, xmax + .1, .1)
    tick_minor = np.arange(xmin, xmax + .01, .02)
    # ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .1))
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    # ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.05))
    # ax[-1].xaxis.set_minor_locator(ticker.LinearLocator(5))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_N_mc_{plot_num+1:02}.png")
    print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_N_mc_{plot_num+1:02}"
    # plt.savefig(f"{ofig}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()


def get_predicted_pick(best_model, best_params, data, sta, tpick_det, opath, plot_num, flag_data, save_plot=True):
    """
    get predicted P- or S-phase onsets
    """
    #
    # apply Monte Carlo Dropout to get predicted time onset with uncertainty (std)
    #
    samp_freq = 100.
    # mc_iter = 50
    mc_iter = 10
    mc_pred = []
    for j in tqdm.tqdm(range(mc_iter)):
        # x_mc = data.reshape(1, data.shape[1], data.shape[2])
        x_mc = data
        y_mc = best_model.predict(x_mc, batch_size=best_params['batch_size'], verbose=0)
        mc_pred.append(y_mc)
    mc_pred = np.array(mc_pred)[:,0,:,:] # mc_pred.shape = (mc_iter, win_size, 1)
    # mc_pred_class = (mc_pred > .5).astype('int32')
    # mc_pred_arg_pick = mc_pred_class.argmax(axis=1)
    # mc_pred_tpick = mc_pred_arg_pick / samp_freq
    # tpick_pred = mc_pred_tpick.mean(axis=0)[0]
    # tpick_pred_std = mc_pred_tpick.std(axis=0)[0]
    # tres = tpick_true - tpick_pred
    #
    mc_pred_mean = mc_pred.mean(axis=0)
    mc_pred_mean_class = (mc_pred_mean > .5).astype('int32')
    mc_pred_mean_arg_pick = mc_pred_mean_class.argmax(axis=0)[0]
    mc_pred_mean_tpick = mc_pred_mean_arg_pick / samp_freq
    mc_pred_std = mc_pred.std(axis=0)
    mc_pred_std_pick = mc_pred_std[mc_pred_mean_arg_pick][0]
    #
    # calculate tpick uncertainty from std of mean probability
    #
    prob_th1 = mc_pred_mean[mc_pred_mean_arg_pick,0] - mc_pred_std_pick
    prob_th2 = mc_pred_mean[mc_pred_mean_arg_pick,0] + mc_pred_std_pick
    cond = (mc_pred_mean > prob_th1) & (mc_pred_mean < prob_th2)
    samps_th = np.arange(mc_pred_mean.shape[0])[cond[:,0]]
    #
    # correction to avoid taking into account samps where mc_pred_mean satisfies cond, but it is not strictly increasing or decresing
    try:
        samps_th1 = np.array([s for s, samp in enumerate(samps_th[:]) if (samp < mc_pred_mean_arg_pick) and (samps_th[s+1] - samp > 1)]).max()
    except ValueError:
        samps_th1 = -1
    try:
        samps_th2 = np.array([s for s, samp in enumerate(samps_th[:-1]) if (samp > mc_pred_mean_arg_pick) and (samps_th[s+1] - samp > 1)]).min()
    except ValueError:
        samps_th2 = len(samps_th)
    #
    samps_th = samps_th[samps_th1+1: samps_th2+1]
    mc_pred_mean_tpick_th1 = samps_th[0] / samp_freq
    mc_pred_mean_tpick_th2 = samps_th[-1] / samp_freq
    # mc_pred_mean_tres = tpick_true - mc_pred_mean_tpick
    #
    # print(tpick_true, tpick_pred, tres, tpick_pred_std)
    # print(tpick_true, mc_pred_mean_tpick, mc_pred_mean_tres, mc_pred_mean_tpick_th1, mc_pred_mean_tpick_th2)
    print(mc_pred_mean_tpick, mc_pred_mean_tpick_th1, mc_pred_mean_tpick_th2)
    #
    # pick class
    #
    # tr_label_3 = f"terr(1 x prob_std) = (-{abs(tpick_pred-tpick_pred_th1):.3f}, +{abs(tpick_pred-tpick_pred_th2):.3f})"
    terr_pre = abs(mc_pred_mean_tpick - mc_pred_mean_tpick_th1)
    terr_pos = abs(mc_pred_mean_tpick - mc_pred_mean_tpick_th2)
    terr_mean = (terr_pre + terr_pos) * .5
    pick_class = 3
    if terr_mean <= .2:
        pick_class -= 1
    if terr_mean <= .1:
        pick_class -= 1
    if terr_mean <= .05:
        pick_class -= 1
    #
    dct_pick = {
        'tpick_det': tpick_det,
        'tpick': mc_pred_mean_tpick,
        'tpick_th1': mc_pred_mean_tpick_th1,
        'tpick_th2': mc_pred_mean_tpick_th2,
        'pick_class': pick_class,
        'terr_pre': terr_pre,
        'terr_pos': terr_pos,
    }
    #
    dct_mc = {
        'mc_pred': mc_pred,
        'mc_pred_mean': mc_pred_mean,
        'mc_pred_mean_class': mc_pred_mean_class,
        'mc_pred_mean_arg_pick': mc_pred_mean_arg_pick,
        'mc_pred_std': mc_pred_std,
        'mc_pred_std_pick': mc_pred_std_pick,
        'prob_th1': prob_th1,
        'prob_th2': prob_th2,
    }
    print(sta, dct_pick)
    #
    if save_plot:
        if data.shape[2] == 1:
            plot_predicted_phase_P(dct_pick, dct_mc, data, sta, tpick_det, opath, plot_num, save_plot)
        elif data.shape[2] == 2:
            plot_predicted_phase_S(dct_pick, dct_mc, data, sta, tpick_det, opath, plot_num, save_plot)
    #
    return dct_pick, dct_mc


def get_predicted_windows(best_params, best_model_pick, best_params_pick, dct_st, dct_param, dct_trigger, flag_data, save_plot=True):
    """
    get predicted P- and S-phase windows, to be used e.g. in change point detection
    """
    #
    tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
    ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
    tstart, tend = dct_st['twin']
    stas = list(dct_st['stt'].keys())
    stas_tp = []
    opath = dct_st['pred'][stas[0]]['opath']
    for sta in stas:
        if len(dct_st['pred'][sta]['p_picks']) == 0:
            continue
        pick_asoc = get_picks_plot(dct_st['pred'][sta], dct_trigger, tp_shift, ts_shift, flag_data)
        print(sta, pick_asoc.keys())
        stas_tp.append((sta, pick_asoc['P']['pick'][0]))
        # p_picks = np.array([tp[0].timestamp for tp in dct_st['pred'][sta]['p_picks']])
        # if len(p_picks) > 0:
        #     stas_tp.append((sta, p_picks.min()))
    #
    stas_tp.sort(key=itemgetter(1))
    for sta_tp in stas_tp:
        print(sta_tp)
    stas_plot = [sp[0] for sp in stas_tp]
    print(stas_plot)
    #
    pick_asoc = {}
    for sta in stas_plot:
        #
        # predicted and detected picks
        print("#")
        #
        p_picks = np.array(dct_st['pred'][sta]['p_picks'])
        s_picks = np.array(dct_st['pred'][sta]['s_picks'])
        pick_asoc[sta] = get_picks_plot(dct_st['pred'][sta], dct_trigger, tp_shift, ts_shift, flag_data)
        #
        twd_exts = {
            "P": [(0., 0.)],
            "S": [(0., 0.)],
        }
        #
        # get P-phase windows
        #
        phase = 'P'
        pick_asoc[sta][phase]['twd'] = {}
        for ii, i in enumerate(pick_asoc[sta][phase]['true_arg']):
            #
            # phase = 'P'
            pick_asoc[sta][phase]['twd'][i] = {}
            # pick = pick_asoc[sta][i][phase]
            trig = pick_asoc[sta][phase]['trig'][i]
            y_prob = dct_st['pred'][sta]['ts'][:,0]
            x_prob = dct_st['pred'][sta]['tt'] + tp_shift
            # tr_win = dct_st['pred'][sta]['tr_win']
            #
            prob_arg = np.argmax(y_prob[trig[0]:trig[1]]) + trig[0]
            # k = prob_pick_arg
            twd_1 = best_params['frac_dsamp_p1'] * best_params['win_size'] * dct_trigger['only_dt']
            twd_2 = best_params['win_size'] * dct_trigger['only_dt'] - twd_1
            #
            for k, twd_ext in enumerate(twd_exts[phase]):
                pick_asoc[sta][phase]['twd'][i][k+1] = {}
                chs = ["N", "E", "Z"]
                for ch in chs:
                    #
                    tr_tmp = dct_st['stt'][sta].select(channel='*'+ch)[0]
                    #
                    # tpick = oc.UTCDateTime(pick['pick']) - tstart_win
                    tstart_win = tr_tmp.stats.starttime + x_prob[prob_arg] - twd_1 - twd_ext[0]
                    tend_win = tr_tmp.stats.starttime + x_prob[prob_arg] + twd_2 + twd_ext[1]
                    pick_asoc[sta][phase]['twd'][i][k+1]['tstart_win'] = tstart_win
                    pick_asoc[sta][phase]['twd'][i][k+1]['tend_win'] = tend_win
                    tpick_win = best_params['frac_dsamp_p1'] * best_params['win_size'] * dct_trigger['only_dt'] + twd_ext[0]
                    pick_asoc[sta][phase]['twd'][i][k+1]['pick_ml_det'] = tpick_win
                    #
                    # processed waveform (input for CNN)
                    #
                    # dct_st_plot[sta][i+1][ch] = tr_tmp.slice(tstart_win, tend_win)
                    pick_asoc[sta][phase]['twd'][i][k+1][ch] = tr_tmp.slice(tstart_win, tend_win)
                    # tr_y = dct_st_plot[sta][i+1][ch].data
                    #
                    # # additional filtered traces for post change-point analysis
                    # #
                    # tr_tmp_2 = dct_st['stt_2'][sta].select(channel='*'+ch)[0]
                    # # dct_st_plot[sta][i+1][f"{ch}_2"] = tr_tmp_2.slice(tstart_win, tend_win)
                    # pick_asoc[sta][i][phase]['twd'][k+1][f"{ch}_2"] = tr_tmp_2.slice(tstart_win, tend_win)
                    #
                    # correct picks: ML phase detection picks --> picks predicted by best picking ML model
                    #
                    # samp_freq = 1 / dct_trigger['only_dt']
                    if ch == 'Z' and best_params_pick['run_mc']:
                        data_P = []
                        data_P.append(pick_asoc[sta][phase]['twd'][i][k+1]['Z'].data[:-1])
                        data_P.append(pick_asoc[sta][phase]['twd'][i][k+1]['E'].data[:-1])
                        data_P.append(pick_asoc[sta][phase]['twd'][i][k+1]['N'].data[:-1])
                        data_P /= np.abs(data_P).max() # normalize before predicting picks
                        data_P = data_P[:1]
                        print("#")
                        print(f"pick: {ii+1}/{len(pick_asoc[sta][phase]['true_arg'])}")
                        print(f"data_P: {data_P.shape}")
                        data_P = data_P.reshape(1, data_P.shape[1], 1)
                        print(f"data_P: {data_P.shape}")
                        print(data_P.mean(), data_P.min(), data_P.max())
                        # pick_asoc[sta][phase]['twd'][i][k+1]['pick_ml'] = get_predicted_pick(best_model_pick['P'], best_params_pick['P'], data_P, sta, tpick_win, opath, ii)
                        dct_pick_ml, dct_mc_ml = get_predicted_pick(best_model_pick['P'], best_params_pick['P'], data_P, sta, tpick_win, opath, ii, flag_data, save_plot)
                        pick_asoc[sta][phase]['twd'][i][k+1]['pick_ml'] = dct_pick_ml
                        pick_asoc[sta][phase]['twd'][i][k+1]['mc_ml'] = dct_mc_ml
                        # best_pred = best_model_pick['P'].predict_classes(data_P, batch_size=best_params_pick['P']['batch_size'], verbose=1)
                        # arg_pick_pred = np.argmax(best_pred[0,:,0])
                        # tpick_pred = arg_pick_pred * dct_trigger['only_dt']
                        # pick_asoc[sta][i][phase]['twd'][k+1]["pick_ml_correc"] = tpick_pred
        #
        # get S-phase windows
        #
        phase = 'S'
        pick_asoc[sta][phase]['twd'] = {}
        # for i in pick_asoc[sta].keys():
        for ii, i in enumerate(pick_asoc[sta][phase]['true_arg']):
            #
            pick_asoc[sta][phase]['twd'][i] = {}
            # pick_asoc[sta][phase]['twd'] = {i: {}}
            trig = pick_asoc[sta][phase]['trig'][i]
            # pick_asoc[sta][i][phase]['twd'] = {}
            # pick = pick_asoc[sta][i][phase]
            # trig = pick['trigs'][j]
            y_prob = dct_st['pred'][sta]['ts'][:,1]
            x_prob = dct_st['pred'][sta]['tt'] + ts_shift
            # tr_win = dct_st['pred'][sta]['tr_win']
            #
            prob_arg = np.argmax(y_prob[trig[0]:trig[1]]) + trig[0]
            twd_1 = best_params['frac_dsamp_s1'] * best_params['win_size'] * dct_trigger['only_dt']
            twd_2 = best_params['win_size'] * dct_trigger['only_dt'] - twd_1
            #
            for k, twd_ext in enumerate(twd_exts[phase]):
                pick_asoc[sta][phase]['twd'][i][k+1] = {}
                # pick_asoc[sta][i][phase]['twd'][k+1] = {}
                for ch in chs:
                    #
                    tr_tmp = dct_st['stt'][sta].select(channel='*'+ch)[0]
                    #
                    # tpick = oc.UTCDateTime(pick['picks'][j]) - tstart_win
                    tstart_win = tr_tmp.stats.starttime + x_prob[prob_arg] - twd_1 - twd_ext[0]
                    tend_win = tr_tmp.stats.starttime + x_prob[prob_arg] + twd_2 + twd_ext[1]
                    pick_asoc[sta][phase]['twd'][i][k+1]['tstart_win'] = tstart_win
                    pick_asoc[sta][phase]['twd'][i][k+1]['tend_win'] = tend_win
                    # pick_asoc[sta][i][phase]['twd'][k+1]['tstart_win'] = tstart_win
                    # pick_asoc[sta][i][phase]['twd'][k+1]['tend_win'] = tend_win
                    tpick_win = best_params['frac_dsamp_s1'] * best_params['win_size'] * dct_trigger['only_dt'] + twd_ext[0]
                    pick_asoc[sta][phase]['twd'][i][k+1]['pick_ml_det'] = tpick_win
                    # pick_asoc[sta][i][phase]['twd'][k+1]["pick_ml_det"] = tpick_win
                    #
                    # processed waveform (input for CNN)
                    #
                    # dct_st_plot[sta][i+1][ch] = tr_tmp.slice(tstart_win, tend_win)
                    pick_asoc[sta][phase]['twd'][i][k+1][ch] = tr_tmp.slice(tstart_win, tend_win)
                    # tr_y = dct_st_plot[sta][i+1][ch].data
                    #
                    # # additional filtered traces for post change-point analysis
                    # #
                    # tr_tmp_2 = dct_st['stt_2'][sta].select(channel='*'+ch)[0]
                    # # dct_st_plot[sta][i+1][f"{ch}_2"] = tr_tmp_2.slice(tstart_win, tend_win)
                    # pick_asoc[sta][i][phase]['twd'][k+1][f"{ch}_2"] = tr_tmp_2.slice(tstart_win, tend_win)
                #
                # correct picks: ML phase detection picks --> picks predicted by best picking ML model
                #
                if best_params_pick['run_mc']:
                    data_S = []
                    data_S.append(pick_asoc[sta][phase]['twd'][i][k+1]['Z'].data[:-1])
                    data_S.append(pick_asoc[sta][phase]['twd'][i][k+1]['E'].data[:-1])
                    data_S.append(pick_asoc[sta][phase]['twd'][i][k+1]['N'].data[:-1])
                    data_S = np.array(data_S)
                    data_S /= np.abs(data_S).max()
                    data_S = data_S[-2:]
                    print("#")
                    print(f"pick: {ii+1}/{len(pick_asoc[sta][phase]['true_arg'])}")
                    print(f"data_S: {data_S.shape}")
                    # data_S = data_S.reshape(1, data_S.shape[1], data_S.shape[0])
                    # data_S = data_S.reshape(1, data_S.shape[1], 2)
                    data_S = data_S.T.reshape(1, data_S.shape[1], 2)
                    print(f"data_S: {data_S.shape}")
                    # print(data_S[0,:,0])
                    print(data_S.mean(), data_S.min(), data_S.max())
                    # pick_asoc[sta][phase]['twd'][i][k+1]['pick_ml'] = get_predicted_pick(best_model_pick['S'], best_params_pick['S'], data_S, sta, tpick_win, opath, ii)
                    dct_pick_ml, dct_mc_ml = get_predicted_pick(best_model_pick['S'], best_params_pick['S'], data_S, sta, tpick_win, opath, ii, flag_data, save_plot)
                    pick_asoc[sta][phase]['twd'][i][k+1]['pick_ml'] = dct_pick_ml
                    pick_asoc[sta][phase]['twd'][i][k+1]['mc_ml'] = dct_mc_ml
                    # best_pred = best_model_pick['S'].predict_classes(data_S, batch_size=best_params_pick['S']['batch_size'], verbose=1)
                    # arg_pick_pred = np.argmax(best_pred[0,:,0])
                    # tpick_pred = arg_pick_pred * dct_trigger['only_dt']
                    # pick_asoc[sta][i][phase]['twd'][k+1]["pick_ml_correc"] = tpick_pred
    #
    # t_str = f"{tstart.year}{tstart.month:02}{tstart.day:02}"
    # t_str += f"T{tstart.hour:02}{tstart.minute:02}{tstart.second:02}"
    # opath = f"{opath}/plots/prediction/events/{flag_data}/ev_{t_str}/plot"
    # os.makedirs(opath, exist_ok=True)
    # ml.export_dict2pckl(pick_asoc, f"{opath}/pick_asoc.pckl")
    #
    return pick_asoc


def save_pick_stats(pick_asoc, dct_st):
    #
    stas = list(dct_st['stt'].keys())
    opath = dct_st['pred'][stas[0]]['opath']
    # opath = f"{opath}/stat_picks"
    # tstart_plot, tend_plot = dct_st['twin']
    # t_str = f"{tstart_plot.year}{tstart_plot.month:02}{tstart_plot.day:02}"
    # t_str += f"T{tstart_plot.hour:02}{tstart_plot.minute:02}{tstart_plot.second:02}"
    # opath = f"{opath}/plots/prediction/events/{flag_data}/ev_{t_str}/picks_ml"
    # os.makedirs(opath, exist_ok=True)
    # ofile_path = f"{opath2}/pick_stats/{s}_pick_detected"
    ofile = open(f"{opath}/pick_stats/pick_refined_stats",'a')
    #
    for sta in pick_asoc:
        #
        for i, k in enumerate(pick_asoc[sta]['P']['true_arg']):
            #
            tpick_det = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['tpick_det']
            tpick_pred = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['tpick']
            terr_pre = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['terr_pre']
            terr_pos = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['terr_pos']
            tpick_th1 = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['tpick_th1']
            tpick_th1 = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['tpick_th2']
            pick_class = pick_asoc[sta]['P']['twd'][k][1]['pick_ml']['pick_class']
            #
            pb_std = pick_asoc[sta]['P']['twd'][k][1]['mc_ml']['mc_pred_std_pick']
            mc_pred_mean = pick_asoc[sta]['P']['twd'][k][1]['mc_ml']['mc_pred_mean']
            mc_pred_mean_arg_pick = pick_asoc[sta]['P']['twd'][k][1]['mc_ml']['mc_pred_mean_arg_pick']
            pb = mc_pred_mean[mc_pred_mean_arg_pick, 0]
            #
            outstr = f"{sta} 'P' {i+1} {tpick_det} {tpick_pred} {terr_pre} {terr_pos} {pick_class} {pb} {pb_std}"
            ofile.write(outstr + '\n')
        #
        for i, k in enumerate(pick_asoc[sta]['S']['true_arg']):
            #
            tpick_det = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['tpick_det']
            tpick_pred = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['tpick']
            terr_pre = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['terr_pre']
            terr_pos = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['terr_pos']
            tpick_th1 = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['tpick_th1']
            tpick_th1 = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['tpick_th2']
            pick_class = pick_asoc[sta]['S']['twd'][k][1]['pick_ml']['pick_class']
            #
            pb_std = pick_asoc[sta]['S']['twd'][k][1]['mc_ml']['mc_pred_std_pick']
            mc_pred_mean = pick_asoc[sta]['S']['twd'][k][1]['mc_ml']['mc_pred_mean']
            mc_pred_mean_arg_pick = pick_asoc[sta]['S']['twd'][k][1]['mc_ml']['mc_pred_mean_arg_pick']
            pb = mc_pred_mean[mc_pred_mean_arg_pick, 0]
            #
            outstr = f"{sta} 'S' {i+1} {tpick_det} {tpick_pred} {terr_pre} {terr_pos} {pick_class} {pb} {pb_std}"
            ofile.write(outstr + '\n')
    #
    ofile.close()


def get_pick_asoc(best_params, best_model_pick, best_params_pick, dct_st, dct_param, dct_trigger, flag_data, save_plot=True, save_stat=True):
    """
    get pre-associated P- and S-phase picks iterating over pre-defined time windows on continuous data
    """
    #
    # get pre-associated picks
    #
    pick_asoc = {}
    for i in dct_st:
        pick_asoc[i] = get_predicted_windows(best_params, best_model_pick, best_params_pick, dct_st[i][1], dct_param, dct_trigger, flag_data, save_plot)
        if save_stat:
            save_pick_stats(pick_asoc[i], dct_st[i][1])
    #
    return pick_asoc


def plot_predicted_wf_phases(best_params, best_model_pick, best_params_pick, ts, dct_st, dct_param, dct_trigger, pick_asoc, opath, flag_data, dct_fmt):
    """
    plot waveforms and predicted picks
    """
    #
    # plot format parameters
    mpl.rcParams['xtick.major.size'] = 14
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 14
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 6
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['ytick.labelsize'] = 22
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['axes.labelsize'] = 22
    #
    print("#")
    #
    stas_plot = list(dct_st[1][1]['stt'].keys())
    print(stas_plot)
    print("#")
    #
    print(list(dct_st.keys()))
    for k in dct_st:
        print(dct_st[k][1]['twin'])
        # print(dct_st[k][1]['stt'])
        print(dct_st[k][1]['stt'][stas_plot[0]].__str__(extended=True))
    #
    # nplot = 0
    print("#")
    print("creating plots...")
    for sta in stas_plot:
        #
        # ncomp = 1
        ch = "Z"
        for i in dct_st:
            #
            # fig = plt.figure(figsize=(20., 10.))
            fig = plt.figure(figsize=(20., 10.))
            plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
            ax = []
            #
            # subplot - processed waveform (input for CNN)
            #
            tr = dct_st[i][1]['stt'][sta].select(channel='*'+ch)[0]
            dt = tr.stats.delta
            tr_y = tr.data
            if dct_param['st_normalized']:
                y_max = np.array([np.abs(tr.data).max() for tr in dct_st[i][1]['stt'][sta]]).max()
                tr_y /= y_max
            tr_x = np.arange(tr.data.size) * dt
            #
            # plot trace
            ax.append(fig.add_subplot(2, 1, 1))
            ax[-1].plot(tr_x, tr_y, c='gray', lw=.25)
            #
            # plot predicted P, S class probability functions
            #
            # ax_tmp = ax[-1].twinx()
            tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
            ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
            #
            x_prob_p = dct_st[i][1]['pred'][sta]['tt']+tp_shift
            y_prob_p = dct_st[i][1]['pred'][sta]['ts'][:,0]
            x_prob_s = dct_st[i][1]['pred'][sta]['tt']+ts_shift
            y_prob_s = dct_st[i][1]['pred'][sta]['ts'][:,1]
            # ax_tmp.plot(x_prob_p, y_prob_p, c='red', lw=0.75)
            # ax_tmp.plot(x_prob_s, y_prob_s, c='blue', lw=0.75)
            # ax_tmp.plot(x_prob_p, y_prob_p, 'ro', ms=1.)
            # ax_tmp.plot(x_prob_s, y_prob_s, 'bo', ms=1.)
            #
            tstart_plot = tr.stats.starttime
            tend_plot = tr.stats.endtime
            print("#")
            print(sta, i, tstart_plot, tend_plot)
            #
            # lines at predicted picks
            #
            for ii, k in enumerate(pick_asoc[i][sta]['P']['true_arg']):
                #
                # P pick corrected after phase picking
                #
                tstart_win = pick_asoc[i][sta]['P']['twd'][k][1]['tstart_win']
                tend_win = pick_asoc[i][sta]['P']['twd'][k][1]['tend_win']
                tpick_pred = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['tpick']
                # tpick_th1 = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['tpick_th1']
                # tpick_th2 = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['tpick_th2']
                # pick_class = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['pick_class']
                tpp_plot_2 = tstart_win - tstart_plot + tpick_pred
                # ax[-1].vlines(x=tpp_plot_2, ymin=-0.05, ymax=1.05, color='r', lw=1., ls='--', clip_on=False)
                ax[-1].axvline(tpp_plot_2, c='r', lw=1.5, ls='-')
            #
            for jj, l in enumerate(pick_asoc[i][sta]['S']['true_arg']):
                #
                # S pick corrected after phase picking
                #
                tstart_win = pick_asoc[i][sta]['S']['twd'][l][1]['tstart_win']
                tpick_pred = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['tpick']
                # tpick_th1 = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['tpick_th1']
                # tpick_th2 = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['tpick_th2']
                # pick_class = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['pick_class']
                tss_plot_2 = tstart_win - tstart_plot + tpick_pred
                # ax[-1].vlines(x=tss_plot_2, ymin=-0.05, ymax=1.05, color='b', lw=1., ls='--', clip_on=False)
                ax[-1].axvline(tss_plot_2, c='b', lw=1.5, ls='-')
            #
            ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,500.)[:])
            # ax[-1].set_ylim([-.5, .5])
            # ax[-1].set_ylim([-.2, .2])
            # ax[-1].set_ylim([-.1, .1])
            # ax[-1].set_ylim([-.05, .05])
            # ax[-1].set_ylim([-.02, .02])
            ylim = dct_fmt[sta]['ylim1']
            ax[-1].set_ylim(ylim)
            # ax[-1].set_xlim([0, ...])
            #
            plt.tight_layout()
            #
            opath = dct_st[i][1]['pred'][sta]['opath']
            tstr = opath.split('wf_')[1].split('/')[0]
            opath2 = f"{opath}/wf_plots"
            os.makedirs(opath2, exist_ok=True)
            #
            ofig = f"{opath2}/fig_pred_{flag_data}_{sta}_{tstr}"
            plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
            plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
            plt.close()


def plot_predicted_wf_phases_prob(best_params, best_model_pick, best_params_pick, ts, dct_st, dct_param, dct_trigger, pick_asoc, opath, flag_data, dct_fmt):
    """
    plot waveforms and predicted picks
    """
    #
    # plot format parameters
    mpl.rcParams['xtick.major.size'] = 14
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 14
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 6
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['ytick.labelsize'] = 22
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['axes.labelsize'] = 22
    #
    print("#")
    #
    stas_plot = list(dct_st[1][1]['stt'].keys())
    print(stas_plot)
    print("#")
    #
    print(list(dct_st.keys()))
    for k in dct_st:
        print(dct_st[k][1]['twin'])
        # print(dct_st[k][1]['stt'])
        print(dct_st[k][1]['stt'][stas_plot[0]].__str__(extended=True))
    #
    # nplot = 0
    print("#")
    print("creating plots...")
    for sta in stas_plot:
        #
        # ncomp = 1
        ch = "Z"
        for i in dct_st:
            #
            # fig = plt.figure(figsize=(20., 10.))
            fig = plt.figure(figsize=(20., 10.))
            plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
            ax = []
            #
            # subplot - processed waveform (input for CNN)
            #
            tr = dct_st[i][1]['stt'][sta].select(channel='*'+ch)[0]
            dt = tr.stats.delta
            tr_y = tr.data
            if dct_param['st_normalized']:
                y_max = np.array([np.abs(tr.data).max() for tr in dct_st[i][1]['stt'][sta]]).max()
                tr_y /= y_max
            tr_x = np.arange(tr.data.size) * dt
            #
            # plot trace
            ax.append(fig.add_subplot(2, 1, 1))
            ax[-1].plot(tr_x, tr_y, c='gray', lw=.25)
            #
            # plot predicted P, S class probability functions
            #
            ax_tmp = ax[-1].twinx()
            tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
            ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * dct_trigger['only_dt']
            #
            x_prob_p = dct_st[i][1]['pred'][sta]['tt']+tp_shift
            y_prob_p = dct_st[i][1]['pred'][sta]['ts'][:,0]
            x_prob_s = dct_st[i][1]['pred'][sta]['tt']+ts_shift
            y_prob_s = dct_st[i][1]['pred'][sta]['ts'][:,1]
            ax_tmp.plot(x_prob_p, y_prob_p, c='red', lw=0.75)
            ax_tmp.plot(x_prob_s, y_prob_s, c='blue', lw=0.75)
            ax_tmp.plot(x_prob_p, y_prob_p, 'ro', ms=1.)
            ax_tmp.plot(x_prob_s, y_prob_s, 'bo', ms=1.)
            #
            tstart_plot = tr.stats.starttime
            tend_plot = tr.stats.endtime
            print("#")
            print(sta, i, tstart_plot, tend_plot)
            #
            # lines at predicted picks
            #
            for ii, k in enumerate(pick_asoc[i][sta]['P']['true_arg']):
                #
                # P pick corrected after phase picking
                #
                tstart_win = pick_asoc[i][sta]['P']['twd'][k][1]['tstart_win']
                tend_win = pick_asoc[i][sta]['P']['twd'][k][1]['tend_win']
                tpick_pred = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['tpick']
                # tpick_th1 = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['tpick_th1']
                # tpick_th2 = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['tpick_th2']
                # pick_class = pick_asoc[i][sta]['P']['twd'][k][1]['pick_ml']['pick_class']
                tpp_plot_2 = tstart_win - tstart_plot + tpick_pred
                # ax[-1].vlines(x=tpp_plot_2, ymin=-0.05, ymax=1.05, color='r', lw=1., ls='--', clip_on=False)
                ax[-1].axvline(tpp_plot_2, c='r', lw=1.5, ls='-')
            #
            for jj, l in enumerate(pick_asoc[i][sta]['S']['true_arg']):
                #
                # S pick corrected after phase picking
                #
                tstart_win = pick_asoc[i][sta]['S']['twd'][l][1]['tstart_win']
                tpick_pred = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['tpick']
                # tpick_th1 = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['tpick_th1']
                # tpick_th2 = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['tpick_th2']
                # pick_class = pick_asoc[i][sta]['S']['twd'][l][1]['pick_ml']['pick_class']
                tss_plot_2 = tstart_win - tstart_plot + tpick_pred
                # ax[-1].vlines(x=tss_plot_2, ymin=-0.05, ymax=1.05, color='b', lw=1., ls='--', clip_on=False)
                ax[-1].axvline(tss_plot_2, c='b', lw=1.5, ls='-')
            #
            # axes properties
            #
            ax_tmp.set_ylim([-0.05, 1.05])
            ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
            #
            ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,500.)[:])
            # ax[-1].set_ylim([-.5, .5])
            # ax[-1].set_ylim([-.2, .2])
            # ax[-1].set_ylim([-.1, .1])
            # ax[-1].set_ylim([-.05, .05])
            # ax[-1].set_ylim([-.02, .02])
            ylim = dct_fmt[sta]['ylim1']
            ax[-1].set_ylim(ylim)
            # ax[-1].set_xlim([0, ...])
            #
            plt.tight_layout()
            #
            opath = dct_st[i][1]['pred'][sta]['opath']
            tstr = opath.split('wf_')[1].split('/')[0]
            opath2 = f"{opath}/wf_plots"
            os.makedirs(opath2, exist_ok=True)
            #
            ofig = f"{opath2}/fig_pred_{flag_data}_{sta}_{tstr}_prob"
            plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
            plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
            plt.close()
