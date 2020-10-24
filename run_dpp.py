#!/usr/bin/env python

import dpp
import obspy.core as oc
import numpy as np
from datetime import datetime
import os, sys, gc


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
dct_det = dpp.get_model_detection(ipath_det, ntrials=1000, verbose=True)
best_model_det = dct_det['best_model']
best_params_det = dct_det['best_params']
best_results_det = dct_det['best_results']
best_hist_det = dct_det['best_hist']
#
# best model for P phase picking
#
ipath_pick_P = "picking/P"
dct_pick_P = dpp.get_model_picking(ipath_pick_P, mode='P', ntrials=50, verbose=True)
best_model_pick_P = dct_pick_P['best_model']
best_params_pick_P = dct_pick_P['best_params']
best_results_pick_P = dct_pick_P['best_results']
best_hist_pick_P = dct_pick_P['best_hist']
#
# best model for S phase picking
#
ipath_pick_S = "picking/S"
dct_pick_S = dpp.get_model_picking(ipath_pick_S, mode='S', ntrials=50, verbose=True)
best_model_pick_S = dct_pick_S['best_model']
best_params_pick_S = dct_pick_S['best_params']
best_results_pick_S = dct_pick_S['best_results']
best_hist_pick_S = dct_pick_S['best_hist']
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
#
#
###### parameters used for detection and picking of phases ######
#
# TODO:
# -> explain these parameters
# -> adapt their names to ones used in paper
#
# dct_param -> dictionary defining how waveform data is to be preprocessed.
# ...: ...
#
dct_param = {
    'st_detrend': True, 'st_filter': False,
    'filter': 'highpass', 'freq_min': .2, 'freq_max': 50.,
    'st_resample': True, 'freq_resample': 100.,
    'start_off': 5., 'stop_off': 0., 'dt_search': 90.,
    'st_normalized': True,
}
#
# dct_trigger -> dictionary defining how predicted probabity time series are
# used to obtain preliminary and refined phase onsets.
# ...: ...
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
###### parameters defining continuous waveform data on which DeepPhasePick is applied ######
#
# TODO: explain parameters in dicts
#
# dct_sta -> dictionary defining the archived waveform data.
# stas: list of stations
# ch: waveform channel code
# net: network code
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
# dct_fmt ->  dictionary defining some formatting for plotting prediction
# results (see functions util_dpp.plot_predicted_wf*).
# ylim1: y-axis limits of plotted seismic trace.
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
# dct_time -> dictionary defining the time over which prediction is made.
# dt_iter: time step between consecutive subwindows
# tstarts: list of starting times of each time window
# tends: list of ending times of each time window
#
dct_time = {
    #
    'tocopilla': {
        'dt_iter': 3600. * 1,
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
        'tstarts': [
            oc.UTCDateTime(2020, 1, 11, 21, 0, 0)
            ],
        'tends': [
            oc.UTCDateTime(2020, 1, 11, 22, 0, 0)
            ],
    },
}
#
#
###### run DeepPhasePick on continuous waveform data ######
#
# flag_data = "tocopilla"
flag_data = "iquique"
# flag_data = "albania"
#
opath = "out"
#
# for i in range(len(tstarts)):
for i in range(len(dct_time[flag_data]['tstarts'])):
    #
    # define time window on continuos seismic waveforms on which prediction is performed
    ts = [dct_time[flag_data]['tstarts'][i], dct_time[flag_data]['tends'][i], dct_time[flag_data]['dt_iter']]
    #
    # perform phase detection: predicting preliminary phase picks
    dct_st = dpp.run_prediction(best_model_det, best_params_det, ts, dct_sta[flag_data], dct_param, dct_trigger, opath, flag_data)
    #
    # perform phase picking: predicting refined phase picks, and optionally plotting them / saving some relevant statistics
    pick_asoc = dpp.get_pick_asoc(best_params_det, best_model_pick, best_params_pick, dct_st, dct_param, dct_trigger, flag_data, save_plot=True, save_stat=True)
    #
    # plot continuous waveform including predicted P, S phases, and corresponding predicted probability time series
    dpp.plot_predicted_wf_phases(best_params_det, best_model_pick, best_params_pick, ts, dct_st, dct_param, dct_trigger, pick_asoc, opath, flag_data, dct_fmt[flag_data])
    dpp.plot_predicted_wf_phases_prob(best_params_det, best_model_pick, best_params_pick, ts, dct_st, dct_param, dct_trigger, pick_asoc, opath, flag_data, dct_fmt[flag_data])
    #
    del dct_st, pick_asoc
    gc.collect()
#
# elapsed time
#
tep_e = datetime.now()
time_exec = tep_e - tep_s
log_out = f"Process completed -- Elapsed time: {time_exec}"
print('')
print(log_out)


