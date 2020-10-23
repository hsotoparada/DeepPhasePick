#!/usr/bin/env python

# import util_soto3, machine_learning3 as ml
import util_dpp as dpp
# import pyasdf
# import obspy.core.event as oc_ev
import obspy.core as oc
import numpy as np
# from glob import glob
# from obspy.signal.trigger import trigger_onset
# from obspy.io.mseed.core import InternalMSEEDError
# import matplotlib as mpl
# import matplotlib.ticker as ticker
# import pylab as plt
from datetime import datetime, timedelta
import calendar
from keras.models import load_model
# from keras import optimizers, utils
# import keras
# import tensorflow as tf
from hyperas.utils import eval_hyperopt_space
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from sklearn.metrics import confusion_matrix
# from sklearn.utils.multiclass import unique_labels
from datetime import datetime
# import itertools
# import tqdm
# from operator import itemgetter
import shutil, os, subprocess, sys, gc


########################################################################################################################
###### MAIN ######
#
tep_s = datetime.now()
#
dpp.init_session()
#
#
###### read optimized models trained for phase detection and picking ######
#
# best model for phase detection
#
ipath_det = "detection"
space_det = dpp.get_space_det()
ntrials = 1000
trials = dpp.import_pckl2dict(f"{ipath_det}/trials_hyperopt_ntrials_{ntrials:003}.pckl")
arg_best_trial = dpp.get_arg_best_trial(trials)
best_trial_det = trials.trials[arg_best_trial]
best_model_det = load_model(f"{ipath_det}/model_hyperopt_t{arg_best_trial:003}.h5")
best_params_det = eval_hyperopt_space(space_det, best_trial_det['misc']['vals'])
best_results_det = dpp.import_pckl2dict(f"{ipath_det}/dict_hyperopt_t{arg_best_trial:003}.pckl")
best_hist_det = best_results_det['history']
print("#")
print(best_model_det.summary())
#
print("######")
print(f"best model for phase detection found for trial {arg_best_trial:03}/{ntrials:003} and hyperparameters:")
for k in best_params_det.keys():
    print(k, best_params_det[k])
print("#")
print(f"best loss for phase detection:")
print(np.array(best_hist_det['val_acc']).max())
# print(best_hist.keys())
#
# best model for P phase picking
#
ipath_pick_P = f"picking/P"
space_pick_P = dpp.get_space_pick_P()
ntrials = 50
trials = dpp.import_pckl2dict(f"{ipath_pick_P}/trials_hyperopt_ntrials_{ntrials:03}.pckl")
arg_best_trial = dpp.get_arg_best_trial(trials)
best_trial_pick_P = trials.trials[arg_best_trial]
best_model_pick_P = load_model(f"{ipath_pick_P}/model_hyperopt_t{arg_best_trial:03}.h5")
best_params_pick_P = eval_hyperopt_space(space_pick_P, best_trial_pick_P['misc']['vals'])
best_results_pick_P = dpp.import_pckl2dict(f"{ipath_pick_P}/dict_hyperopt_t{arg_best_trial:03}.pckl")
best_hist_pick_P = best_results_pick_P['history']
print("#")
print(best_model_pick_P.summary())
#
print("######")
print(f"best model for P phase picking found for trial {arg_best_trial:03}/{ntrials:03} and hyperparameters:")
for k in best_params_pick_P.keys():
    print(k, best_params_pick_P[k])
print("#")
print(f"best val acc for P phase picking:")
print(np.array(best_hist_pick_P['val_acc']).max())
# print(best_hist_pick_P.keys())
# sys.exit()
#
# best model for S phase picking
#
ipath_pick_S = f"picking/S"
space_pick_S = dpp.get_space_pick_S()
ntrials = 50
trials = dpp.import_pckl2dict(f"{ipath_pick_S}/trials_hyperopt_ntrials_{ntrials:03}.pckl")
arg_best_trial = dpp.get_arg_best_trial(trials)
best_trial_pick_S = trials.trials[arg_best_trial]
best_model_pick_S = load_model(f"{ipath_pick_S}/model_hyperopt_t{arg_best_trial:03}.h5")
best_params_pick_S = eval_hyperopt_space(space_pick_S, best_trial_pick_S['misc']['vals'])
best_results_pick_S = dpp.import_pckl2dict(f"{ipath_pick_S}/dict_hyperopt_t{arg_best_trial:03}.pckl")
best_hist_pick_S = best_results_pick_S['history']
print("#")
print(best_model_pick_S.summary())
#
print("######")
print(f"best model for S phase picking found for trial {arg_best_trial:03}/{ntrials:03} and hyperparameters:")
for k in best_params_pick_S.keys():
    print(k, best_params_pick_S[k])
print("#")
print(f"best val acc for S phase picking:")
print(np.array(best_hist_pick_S['val_acc']).max())
# print(type(best_hist_pick_S['val_acc']))
#
# best models and parameters for phase picking
#
best_model_pick = {
    'P': best_model_pick_P,
    'S': best_model_pick_S,
}
best_params_pick = {
    'P': best_params_pick_P,
    'S': best_params_pick_S,
    'run_mc': True,
    # 'run_mc': False,
}
# sys.exit()
#
#
###### global parameters for predicted windows ######
#
# TODO:
# -> explain these parameters
# -> adapt their names to ones used in paper
#
phase_names = {
    0: "P", 1: "S", 2: "N"
}
dct_param = {
    'st_detrend': True, 'st_filter': False,
    'filter': 'highpass', 'freq_min': .2, 'freq_max': 50.,
    'st_resample': True, 'freq_resample': 100.,
    'start_off': 5., 'stop_off': 0., 'dt_search': 90.,
    'st_normalized': True,
}
#
dct_trigger = {
    'only_dt': 0.01, 'n_shift': 10, 'pthres_p': [0.98, .001], 'pthres_s': [0.98, .001], 'max_trig_len': [9e99, 9e99],
    # 'only_dt': 0.01, 'n_shift': 10, 'pthres_p': [0.95, .001], 'pthres_s': [0.95, .001], 'max_trig_len': [9e99, 9e99],
}
dct_trigger['half_dur'] = best_params_det['win_size'] * dct_trigger['only_dt'] *.5
dct_trigger['batch_size'] = best_params_det['batch_size']
dct_trigger['n_win'] = int(dct_trigger['half_dur']/dct_trigger['only_dt'])
dct_trigger['n_feat'] = 2 * dct_trigger['n_win']
#
dct_trigger['detec_thres'] = {
    #
    'tocopilla':{
        'dt_PS_max': 35.,
        'dt_max_s_dup': 3.,
        'dt_max_sp_near': 3.,
        'pb_thres': .5,
        'tp_th_add': 1.5,
    },
    #
    'iquique':{
        'dt_PS_max': 35.,
        'dt_max_s_dup': 3.,
        'dt_max_sp_near': 3.,
        'pb_thres': .5,
        'tp_th_add': 1.5,
    },
    #
    'albania':{
        'dt_PS_max': 25.,
        'tp_th_add': 1.5,
        'dt_max_s_dup': 2.,
        'dt_max_sp_near': 1.,
        'pb_thres': .5,
    },
}
#
print("######")
for k in dct_trigger.keys():
    print(k, dct_trigger[k])
#
#
###### select continuous waveforms on which prediction will be made  ######
#
# 2007 M7.7 Tocopilla earthquake at 2007-11-14T15:40:50.050000Z
#
# stas = ['PB06']
# dct_sta = {
#     'stas': stas,
#     'ch': 'HH',
#     'net': 'CX',
# }
# dct_fmt = {
#     'PB06':{
#         'ylim1': [-.02, .02],
#     },
# }
# dt_iter = 3600. * 1.
# dt_win = dt_iter
# flag_data = "tocopilla"
#
# tstarts = [
#     oc.UTCDateTime(2007, 11, 20, 7, 0, 0)
# ]
# #
# tends = [
#     oc.UTCDateTime(2007, 11, 20, 8, 0, 0)
# ]
#
# 2014 M8.1 Iquique earthquake at 2014-04-01T23:46:45.720000Z
# -> largest foreshock: M6.6 at 2014-03-16T21:16:28
# -> largest aftershock: M7.6 at 2014-04-03T02:43:13
#
# stas = ['PB01']
# dct_sta = {
#     'stas': stas,
#     'ch': 'HH',
#     'net': 'CX',
# }
# dct_fmt = {
#     'PB01':{
#         'ylim1': 0.02,
#     },
# }
# dt_iter = 3600. * 1. # in seconds
# dt_win = dt_iter # in seconds
# flag_data = "iquique"
#
# tstarts = [
#     # oc.UTCDateTime(2014, 4, 3, 2, 0, 0)
#     oc.UTCDateTime(2014, 4, 3, 2, 8, 0)
# ]
# #
# tends = [
#     # oc.UTCDateTime(2014, 4, 3, 3, 0, 0)
#     oc.UTCDateTime(2014, 4, 3, 3, 8, 0)
# ]
#
# 2019 M6.4 Albania earthquake at 2019-11-26T02:54:11.30000Z
#
# stas = ['AB21']
# dct_sta = {
#     'stas': stas,
#     'ch': 'HH',
#     'net': '9K',
# }
# dct_fmt = {
#     'AB21':{
#         'ylim1': [-.2, .2],
#     },
# }
# dt_iter = 3600. * 1.
# dt_win = dt_iter
# flag_data = "albania"
#
# tstarts = [
#     oc.UTCDateTime(2020, 1, 11, 21, 0, 0)
# ]
# #
# tends = [
#     oc.UTCDateTime(2020, 1, 11, 22, 0, 0)
# ]
#
#
###### run prediction on waveform data ######
#
# TODO: explain parameters in dicts
#
dct_sta = {
    #
    'tocopilla': {
        'stas': ['PB06'],
        'ch': 'HH',
        'net': 'CX',
    },
    #
    'iquique': {
        'stas': ['PB01'],
        'ch': 'HH',
        'net': 'CX',
    },
    #
    'albania': {
        'stas': ['AB21'],
        'ch': 'HH',
        'net': '9K',
    },
}
#
dct_fmt = {
    #
    'tocopilla': {
        'PB06':{
            'ylim1': [-.02, .02],
        },
    },
    #
    'iquique': {
        'PB01':{
            'ylim1': [-0.02, 0.02],
        },
    },
    #
    'albania': {
        'AB21':{
            'ylim1': [-.2, .2],
        },
    },
}
#
dct_time = {
    #
    'tocopilla': {
        'dt_iter': 3600. * 1,
        'dt_win': 3600. * 1,
        'tstarts': [
            oc.UTCDateTime(2007, 11, 20, 7, 0, 0)
            ],
        'tends': [
            oc.UTCDateTime(2007, 11, 20, 8, 0, 0)
            ],
    },
    #
    'iquique': {
        'dt_iter': 3600. * 1,
        'dt_win': 3600. * 1,
        'tstarts': [
            # oc.UTCDateTime(2014, 4, 3, 2, 0, 0)
            oc.UTCDateTime(2014, 4, 3, 2, 8, 0)
            ],
        'tends': [
            # oc.UTCDateTime(2014, 4, 3, 3, 0, 0)
            oc.UTCDateTime(2014, 4, 3, 3, 8, 0)
            ],
    },
    #
    'albania': {
        'dt_iter': 3600. * 1,
        'dt_win': 3600. * 1,
        'tstarts': [
            oc.UTCDateTime(2020, 1, 11, 21, 0, 0)
            ],
        'tends': [
            oc.UTCDateTime(2020, 1, 11, 22, 0, 0)
            ],
    },
}
#
flag_data = "tocopilla"
# flag_data = "iquique"
# flag_data = "albania"
#
opath = "out"
#
# for i in range(len(tstarts)):
for i in range(len(dct_time[flag_data]['tstarts'])):
    #
    # perform phase detection: predicting preliminary phase picks
    #
    # ts = [tstarts[i], tends[i], dt_iter, dt_win]
    ts = [dct_time[flag_data]['tstarts'][i], dct_time[flag_data]['tends'][i], dct_time[flag_data]['dt_iter'], dct_time[flag_data]['dt_win']]
    # print("ts:")
    # print(ts)
    # sys.exit()
    #
    dct_st = dpp.run_prediction(best_model_det, best_params_det, ts, dct_sta[flag_data], dct_param, dct_trigger, opath, flag_data)
    #
    # perform phase picking: predicting refined phase picks, and optionally plotting them and saving some relevant statistics
    #
    pick_asoc = dpp.get_pick_asoc(best_params_det, best_model_pick, best_params_pick, dct_st, dct_param, dct_trigger, flag_data, save_plot=True, save_stat=True)
    #
    # plot continuous waveform including predicted P, S phases, and corresponding predicted probability time series
    #
    dpp.plot_predicted_wf_phases(best_params_det, best_model_pick, best_params_pick, ts, dct_st, dct_param, dct_trigger, pick_asoc, opath, flag_data, dct_fmt[flag_data])
    dpp.plot_predicted_wf_phases_prob(best_params_det, best_model_pick, best_params_pick, ts, dct_st, dct_param, dct_trigger, pick_asoc, opath, flag_data, dct_fmt[flag_data])
    #
    del dct_st, pick_asoc
    gc.collect()
#
# elapsed time
tep_e = datetime.now()
time_exec = tep_e - tep_s
log_out = f"Process completed -- Elapsed time: {time_exec}"
print('')
print(log_out)


