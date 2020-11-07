#!/usr/bin/env python
#
# This scripts runs DeepPhasePick, a method for automatically detecting and picking seismic phases from local earthquakes based on highly optimized deep neural networks.
# For more info, see the github repository:
#
# https://github.com/hsotoparada/DeepPhasePick
#
# and the original publication:
#
# Soto and Schurr (2020). DeepPhasePick: A method for Detecting and Picking Seismic Phases from Local Earthquakes
# based on highly optimized Convolutional and Recurrent Deep Neural Networks.
# https://eartharxiv.org/repository/view/1752/
#
# Author: Hugo Soto Parada (2020)
# Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com

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
###### Reading of optimized models trained for phase detection and picking tasks ######
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
}
#
#
###### Definition of user-defined parameters used for detection and picking of phases ######
#
# -----
# dct_param -> dictionary defining how waveform data is to be preprocessed.
# -----
# samp_freq: (float) sampling rate (in hertz) of the seismic waveforms. Waveforms with different sampling rate should be resampled to samp_freq.
# samp_dt: (float) sample distance (in seconds) of the seismic waveforms. Defined as 1/samp_freq
# st_normalized: (bool) True to normalize waveforms on which phase detection will be performed.
# st_detrend: (bool) True to detrend (linear) waveforms on which phase detection will be performed.
# st_filter: (bool) True to filter waveforms on which phase detection will be performed.
# filter: (str) type of filter applied to waveforms on which phase detection will be performed.
# filter_freq_min: (float) minimum frequency of filter applied. Data below this frequency is removed.
# filter_freq_max: (float) maximum frequency of filter applied. Data above this frequency is removed.
#
samp_freq = 100.
dct_param = {
    'samp_freq': samp_freq,
    'samp_dt': 1 / samp_freq,
    'st_normalized': True,
    'st_detrend': True,
    'st_filter': False,
    'filter': 'highpass',
    # 'filter': 'bandpass',
    'filter_freq_min': .2,
    'filter_freq_max': 10.,
}
#
# -----
# dct_trigger -> dictionary defining how predicted discrete probability time series are obtained and used to obtain preliminary and refined phase onsets.
# -----
# detec_thres: dictionary containing user-defined parameters defining how the preliminary onsets are obtained in the phase detection stage.
#   n_shift: (int) step size (in samples) defining discrete probability time series.
#   pthres_p: (list) probability thresholds defining P-phase trigger on (pthres_p[0]) and off (pthres_p[1]) times, as thres1 and thres2 parameters in obspy trigger_onset function.
#   pthres_s: (list) probability thresholds defining S-phase trigger on (pthres_s[0]) and off (pthres_s[1]) times, as thres1 and thres2 parameters in obspy trigger_onset function.
#   max_trig_len: (list) maximum lengths (in samples) of triggered P (max_trig_len[0]) and S (max_trig_len[1]) phase, as max_len parameter in obspy trigger_onset function.
#
# detec_cond: dictionary containing user-defined parameters applied in optional conditions for improving phase detections, by keeping or removing presumed false preliminary onsets.
# These conditions are explained in Supplementary Information of the original publication (see https://eartharxiv.org/repository/view/1752/).
#   op_conds: (list) optional conditions that will be applied. For example ['1', '2'] indicates that only conditions (1) and (2) will be used.
#   dt_PS_max: (float) time (in seconds) used to define search time intervals in conditions (1) and (2).
#   dt_sdup_max: (float) time threshold (in seconds) used in condition (3).
#   dt_sp_near: (float) time threshold (in seconds) used in condition (4).
#   tp_th_add: (float) time (in seconds) added to dt_PS_max to define search time intervals in condition (1).
#
# mcd: dictionary containing user-defined parameters controlling how Monte Carlo Dropout MCD technique is applied in the phase picking stage.
#   run_mcd: (bool) True to actually run phase picking stage in order to refine preliminary picks from phase detection.
#   mcd_iter: (int) number of MCD iterations used.
#
dct_trigger = {}
#
dct_trigger['det_thres'] = {
    'n_shift': 10, 'pthres_p': [0.98, .001], 'pthres_s': [0.98, .001], 'max_trig_len': [9e99, 9e99],
    # 'n_shift': 10, 'pthres_p': [0.95, .001], 'pthres_s': [0.95, .001], 'max_trig_len': [9e99, 9e99],
}
#
dct_trigger['det_cond'] = {
    'op_conds': ['1', '2', '3', '4'],
    'dt_PS_max': 35.,
    'dt_sdup_max': 3.,
    'dt_sp_near': 3.,
    'tp_th_add': 1.5,
}
#
dct_trigger['mcd'] = {
    'run_mcd': True,
    'mcd_iter': 10,
}
#
print("######")
for k in dct_trigger.keys():
    print(k, dct_trigger[k])
#
#
###### Parameters defining continuous waveform data on which DeepPhasePick is applied ######
#
# -----
# dct_sta -> dictionary defining the archived waveform data on which the method will be applied.
# -----
# stas: (list) stations.
# ch: (str) waveform channel code.
# net: (str) network code.
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
}
#
# -----
# dct_fmt -> dictionary defining some formatting options for plotting prediction results (see functions util_dpp.plot_predicted_wf*).
# -----
# ylim1: (list) y-axis limits of plotted seismic trace.
# dx_tick: (float) x-axis ticks spacing in plotted seismic trace.
#
dct_fmt = {
    #
    'tocopilla': {
        'PB06':{
            'ylim1': [-.02, .02],
            'dx_tick': 500.,
        },
    },
    #
    'iquique': {
        'PB01':{
            'ylim1': [-0.02, 0.02],
            'dx_tick': 500.,
        },
    },
}
#
# -----
# dct_time -> dictionary defining time windows over which predictions are made.
# -----
# dt_iter: (float) time step (in seconds) between consecutive time windows.
# tstarts: (list) starting times of each time window.
# tends: (list) ending times of each time window.
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
}
#
#
###### Run DeepPhasePick on continuous waveform data ######
#
# flag_data = "tocopilla"
flag_data = "iquique"
#
opath = "out"
#
for i in range(len(dct_time[flag_data]['tstarts'])):
    #
    # define time window on continuos seismic waveforms on which prediction is performed
    pred_times = [dct_time[flag_data]['tstarts'][i], dct_time[flag_data]['tends'][i], dct_time[flag_data]['dt_iter']]
    #
    # perform phase detection: prediction of preliminary phase picks
    dct_dets = dpp.run_detection(best_model_det, best_params_det, pred_times, dct_sta[flag_data], dct_param, dct_trigger, opath, flag_data)
    #
    # perform phase picking: prediction of refined phase picks, and optionally plotting them and saving some relevant statistics
    dct_picks = dpp.run_picking(best_params_det, best_model_pick, best_params_pick, dct_dets, dct_param, dct_trigger, flag_data, save_plot=True, save_stat=True)
    #
    # plotting of continuous waveform with predicted P and S phases, and corresponding predicted probability time series
    dpp.plot_predicted_wf_phases(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, dct_fmt[flag_data], comp="E")
    # dpp.plot_predicted_wf_phases(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, dct_fmt[flag_data], comp="N")
    dpp.plot_predicted_wf_phases(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, dct_fmt[flag_data], comp="Z")
    dpp.plot_predicted_wf_phases_prob(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, dct_fmt[flag_data], comp="E")
    # dpp.plot_predicted_wf_phases_prob(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, dct_fmt[flag_data], comp="N")
    dpp.plot_predicted_wf_phases_prob(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, dct_fmt[flag_data], comp="Z")
    #
    # dpp.plotly_predicted_wf_phases(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, comps=["Z","E","N"])
    dpp.plotly_predicted_wf_phases(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, comps=["Z","N"])
    # dpp.plotly_predicted_wf_phases(best_params_det, dct_dets, dct_param, dct_trigger, dct_picks, flag_data, comps=["Z"])
    #
    del dct_dets, dct_picks
    gc.collect()
#
# elapsed time
#
tep_e = datetime.now()
time_exec = tep_e - tep_s
log_out = f"Process completed -- Elapsed time: {time_exec}"
print('')
print(log_out)


