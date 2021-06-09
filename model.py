#module dpp, for python 3.x
#coding=utf-8
#
# This module contains functions defining DeepPhasePick, a method for automatically detecting and picking seismic phases
# from local earthquakes based on highly optimized deep neural networks.
# For more info, see the github repository:
#
# https://github.com/hsotoparada/DeepPhasePick
#
# and the original publication:
#
# Soto and Schurr (2020).
# DeepPhasePick: A method for Detecting and Picking Seismic Phases from Local Earthquakes
# based on highly optimized Convolutional and Recurrent Deep Neural Networks.
# https://eartharxiv.org/repository/view/1752/
#
# Author: Hugo Soto Parada (2020)
# Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com

import util
import numpy as np
from obspy.signal.trigger import trigger_onset
from obspy.io.mseed.core import InternalMSEEDError
from keras.models import load_model
import tqdm
import re, sys, os, shutil, gc


class Model():
    """
    Model stuff...
    """

    # TODO: maybe it's better to define models directly from their complete paths
    def __init__(
        self,
        path_det="detection",
        ntrials_det=1000,
        path_pick_P="picking/P",
        ntrials_P=50,
        path_pick_S="picking/S",
        ntrials_S=50
    ):

        self.path_det = path_det
        self.path_pick_P = path_pick_P
        self.path_pick_S = path_pick_S
        self.ntrials_det = ntrials_det
        self.ntrials_P = ntrials_P
        self.ntrials_S = ntrials_S

        self.model_detection = self._get_model_detection(verbose=True)
        self.model_picking_P = self._get_model_picking(mode='P', verbose=True)
        self.model_picking_S = self._get_model_picking(mode='S', verbose=True)


    # def get_model_detection(ipath, ntrials, verbose=True):
    def _get_model_detection(self, verbose=True):
        """
        Returns dictionary with best model and other relevant results obtained from
        hyperparameter optimization performed for the phase detection task.
        ----------
        ipath: (str) path to input trained detection model.
        ntrials: (int) number of trials run during the hyperparameter optimization of the task at hand.
        """
        #
        ipath = self.path_det
        space = util.import_pckl2dict(f"{ipath}/space_detection.pckl") #TODO: error -> ModuleNotFoundError: No module named 'hyperopt.pyll'; 'hyperopt' is not a package
        trials = util.import_pckl2dict(f"{ipath}/trials_hyperopt_ntrials_{self.ntrials_det:03}.pckl")
        arg_best_trial = util.get_arg_best_trial(trials)
        best_results = util.import_pckl2dict(f"{ipath}/dict_hyperopt_t{arg_best_trial:03}.pckl")
        best_params = best_results['params']
        best_model = load_model(f"{ipath}/model_hyperopt_t{arg_best_trial:03}.h5")
        best_hist = best_results['history']
        # best_hist = util.import_pckl2dict(f"{ipath}/hist_model_hyperopt_t{arg_best_trial:03}.pckl")
        #
        if verbose:
            print("#")
            print(best_model.summary())
            #
            print("######")
            print(f"best model for phase detection found for trial {arg_best_trial:03}/{self.ntrials_det:03} and hyperparameters:")
            for k in best_params:
                print(k, best_params[k])
            print("#")
            print(f"best acc for phase detection:")
            print(np.array(best_hist['val_acc']).max())
            # print(np.array(best_hist['acc']).max())
        #
        dct = {
            'best_model': best_model,
            'best_params': best_params,
            'best_hist': best_hist,
        }
        #
        return dct


    # def get_model_picking(ipath, mode, ntrials, verbose=True):
    def _get_model_picking(self, mode, verbose=True):
        """
        Returns dictionary with best model and other relevant results obtained from
        hyperparameter optimization performed for the phase picking task.
        ----------
        ipath: (str) path to input trained picking model.
        mode: (str) 'P' or 'S' for retrieving P- or S- phase picking model, respectively.
        ntrials: (int) number of trials run during the hyperparameter optimization of the task at hand.
        """
        #
        if mode == 'P':
            ipath = self.path_pick_P
            ntrials = self.ntrials_P
            space = util.import_pckl2dict(f"{ipath}/space_picking_P.pckl")
        else:
            ipath = self.path_pick_S
            ntrials = self.ntrials_S
            space = util.import_pckl2dict(f"{ipath}/space_picking_S.pckl")
        #
        trials = util.import_pckl2dict(f"{ipath}/trials_hyperopt_ntrials_{ntrials:03}.pckl")
        arg_best_trial = util.get_arg_best_trial(trials)
        best_results = util.import_pckl2dict(f"{ipath}/dict_hyperopt_t{arg_best_trial:03}.pckl")
        best_params = best_results['params']
        best_model = load_model(f"{ipath}/model_hyperopt_t{arg_best_trial:03}.h5")
        best_hist = best_results['history']
        # best_hist = util.import_pckl2dict(f"{ipath}/hist_model_hyperopt_t{arg_best_trial:03}.pckl")
        #
        if verbose:
            print("#")
            print(best_model.summary())
            #
            print("######")
            print(f"best model for {mode} phase picking found for trial {arg_best_trial:03}/{ntrials:03} and hyperparameters:")
            for k in best_params:
                print(k, best_params[k])
            print("#")
            print(f"best acc for {mode} phase picking:")
            # print(np.array(best_hist['val_acc']).max())
            print(np.array(best_hist['acc']).max())
        #
        dct = {
            'best_model': best_model,
            'best_params': best_params,
            'best_hist': best_hist,
        }
        #
        return dct


    # def sliding_window(data, size, stepsize=1, axis=-1):
    def _sliding_window(self, data, size, stepsize=1, axis=-1):
        """
        Adapted from similar function in https://github.com/interseismic/generalized-phase-detection (see Ross et al., 2018; doi:10.1785/0120180080)
        Calculates a sliding window over data.
        Returns a numpy array where rows are instances of the sliding window.
        ----------
        data: (numpy array) data to be slided over.
        size: (int) sliding window size.
        stepsize: (int) sliding window stepsize.
        axis: (int) axis to slide over. Defaults to the last axis.
        ----------
        Examples
        ----------
        >>> a = numpy.array([1, 2, 3, 4, 5])
        >>> sliding_window(a, size=3)
        array([[1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]])
        >>> sliding_window(a, size=3, stepsize=2)
        array([[1, 2, 3],
            [3, 4, 5]])
        """
        if axis >= data.ndim:
            raise ValueError(
                "Axis value out of range"
            )
        #
        stepsize = int(stepsize)
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
        return strided.copy()


                # tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = self._make_prediction(config, model, st_tmp)
                # # tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = make_prediction(best_model, best_params, st_tmp, dct_param, dct_trigger)
    # def make_prediction(best_model, best_params, st, dct_param, dct_trigger):
    def _make_prediction(self, config, model, st):
        """
        Applies best trained detection model on waveform data and returns predicted P- and S-phase discrete probability time series.
        ----------
        best_model: best (keras) model trained for phase detection.
        best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
        st: (obspy stream) seismic stream on which predictions are made.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        """
        #
        # trim traces within common start and end times to avoid this type of exception:
        # --> ValueError: could not broadcast input array from shape (17958,460) into shape (17963,460)
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
        # Reshaping data matrix for sliding window
        # print("Reshaping data matrix for sliding window")
        #
        st_data = [st[0].data, st[1].data, st[2].data]
        if config.params['st_normalized']:
            data_max = np.array([np.abs(tr.data).max() for tr in st]).max()
            for i, tr_data in enumerate(st_data):
                tr_data /= data_max
        #
        best_params = model.model_detection['best_params']
        best_model = model.model_detection['best_model']
        dt = st[0].stats.delta
        tt = (np.arange(0, st_data[0].size, config.trigger['n_shift']) + .5 * best_params['win_size']) * dt #[sec]
        #
        try:
            sliding_N = self._sliding_window(st_data[0], best_params['win_size'], stepsize=config.trigger['n_shift'])
            sliding_E = self._sliding_window(st_data[1], best_params['win_size'], stepsize=config.trigger['n_shift'])
            sliding_Z = self._sliding_window(st_data[2], best_params['win_size'], stepsize=config.trigger['n_shift'])
            tr_win = np.zeros((sliding_N.shape[0], best_params['win_size'], 3))
            tr_win[:,:,0] = sliding_N
            tr_win[:,:,1] = sliding_E
            tr_win[:,:,2] = sliding_Z
            #
            # normalization, in separated operations to avoid memory errors
            aa = np.abs(tr_win)
            bb = np.max(aa, axis=(1,2))[:,None,None]
            tr_win = tr_win / bb
            #tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
            #
            tt = tt[:tr_win.shape[0]]
            #
            # make model predictions
            ts = best_model.predict(tr_win, verbose=True, batch_size=best_params['batch_size'])
            #
            prob_P = ts[:,0]
            prob_S = ts[:,1]
            prob_N = ts[:,2]
        except ValueError:
            tt, ts, prob_S, prob_P, prob_N = [0],[0],[0],[0],[0]
        #     print(tt.shape, ts.shape, prob_S.shape, prob_P.shape, prob_N.shape)
        #
        return (tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st)


                # p_picks, s_picks, p_trigs, s_trigs = self._calculate_trigger(config, model, st_tmp, net, s, tt, ts, prob_P, prob_S)
                # # p_picks, s_picks, p_trigs, s_trigs = calculate_trigger(st_tmp, net, s, tt, ts, prob_P, prob_S, dct_param, dct_trigger, best_params)
    # def calculate_trigger(st, net, sta, tt, ts, prob_P, prob_S, dct_param, dct_trigger, best_params):
    def _calculate_trigger(self, config, model, st, net, sta, tt, ts, prob_P, prob_S):
        """
        Calculates trigger on and off times of P- and S-phases, from predicted discrete P,S-class probability time series.
        Returns preliminary phase onset and trigger on/off times.
        ----------
        st: seismic stream on which predictions were made and trigger times are calculated.
        net: (str) network code of seismic stream.
        sta: (str) station code of seismic stream.
        tt: (numpy array) array of times within seismic stream to which discrete probabilities are assigned (centers of sliding windows).
        ts: (numpy array) array containing discrete probability time series of predicted P, S, and N classes.
        prob_P: (numpy array) discrete probability time series of predicted P class.
        prob_S: (numpy array) discrete probability time series of predicted S class.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
        """
        #
        #trigger_onset(charfct, thres1, thres2, max_len=9e+99, max_len_delete=False)
        #calculates trigger on and off times from characteristic function charfct, given thresholds thres1 and thres2
        #
        #correction of time position for P, S predicted picks
        #
        best_params = model.model_detection['best_params']
        samp_dt = 1 / config.params['samp_freq']
        tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
        ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
        # tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * config.params['samp_dt']
        # ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * config.params['samp_dt']
        #
        #calculate trigger on and off times of P phases, from predicted P-class probability
        #
        p_picks = []
        p_trigs = trigger_onset(prob_P, config.trigger['pthres_p'][0], config.trigger['pthres_p'][1], config.trigger['max_trig_len'][0])
        for trig in p_trigs:
            if trig[1] == trig[0]:
                continue
            pick = np.argmax(ts[trig[0]:trig[1], 0])+trig[0]
            stamp_pick = st[0].stats.starttime + tt[pick] + tp_shift
            # print(pick, tt[pick])
            # print(f"P {pick} {ts[pick][0]} {stamp_pick} {tp_shift:.2f}")
            p_picks.append((stamp_pick, pick))
        #
        #calculate trigger on and off times of S phases, from predicted S-class probability
        #
        s_picks = []
        s_trigs = trigger_onset(prob_S, config.trigger['pthres_s'][0], config.trigger['pthres_s'][1], config.trigger['max_trig_len'][1])
        for trig in s_trigs:
            if trig[1] == trig[0]:
                continue
            pick = np.argmax(ts[trig[0]:trig[1], 1])+trig[0]
            stamp_pick = st[0].stats.starttime + tt[pick] + ts_shift
            # print(f"S {pick} {ts[pick][1]} {stamp_pick} {ts_shift:.2f}")
            s_picks.append((stamp_pick, pick))
        #
        return p_picks, s_picks, p_trigs, s_trigs


    # TODO: create method: read_detections. To read saved detections dict into model object
    # def run_detection(best_model, best_params, dct_data, dct_param, dct_trigger, dct_out):
    def run_detection(self, config, data, model):
        """
        Performs P- and S-phase detection task.
        Returns a dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
        ----------
        best_model: best (keras) model trained for phase detection.
        best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
        dct_time: (list) dictionary defining time windows over which prediction is performed.
        dct_sta: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        dct_out: (dict) dictionary defining DeepPhasePick output options.
        t: (int) index number of the waveform time window on which prediction is performed.
        """
        #
        # detect seismic phases on processed stream data
        #
        # dct_dets = {}
        self.detections = {}
        # for i in dct_data:
        for i in data.data:
            self.detections[i] = {'pred': {}}
            for s in sorted(data.data[i]['st'].keys())[:]:
                #
                st_tmp = data.data[i]['st'][s]
                net = st_tmp[0].stats.network
                ch = st_tmp[0].stats.channel
                dt = st_tmp[0].stats.delta
                print("#")
                print(f"Calculating predictions for stream: {net}.{s}..{ch[:-1]}?...")
                print(st_tmp)
                #
                # get predicted discrete phase class probability time series
                #
                tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = self._make_prediction(config, model, st_tmp)
                # tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = make_prediction(best_model, best_params, st_tmp, dct_param, dct_trigger)
                # print(tt.shape, ts.shape, prob_S.shape, prob_P.shape, prob_N.shape)
                #
                # skip streams raising ValueError in make_prediction()
                if len(ts) == 1 and ts[0] == 0:
                    val_err = f"Sliding window size may not exceed size of selected axis"
                    print(f"skipping stream {net}.{s}..{ch[:-1]}? due to ValueError: {val_err}...")
                    continue
                #
                # get preliminary phase picks
                #
                opath = data.data[i]['opath']
                os.makedirs(f"{opath}/pick_stats", exist_ok=True)
                # ofile_path = f"{opath}/pick_stats/{s}_pick_detected"
                p_picks, s_picks, p_trigs, s_trigs = self._calculate_trigger(config, model, st_tmp, net, s, tt, ts, prob_P, prob_S)
                # p_picks, s_picks, p_trigs, s_trigs = calculate_trigger(st_tmp, net, s, tt, ts, prob_P, prob_S, dct_param, dct_trigger, best_params)
                print(f"p_picks = {len(p_picks)}, s_picks = {len(s_picks)}")
                #
                self.detections[i]['pred'][s] = {
                    'dt': dt, 'tt': tt, 'ts': ts,
                    'p_picks': p_picks, 's_picks': s_picks,
                    'p_trigs': p_trigs, 's_trigs': s_trigs,
                    'opath': opath,
                }
                # TODO: check if this actually update the data object
                if st_trim_flag:
                    data.data[i]['st'][s] = st_trimmed
        #
        # return self.detections, data


                # # dct_picks[k][sta] = get_dct_picks(self.detections[k]['pred'][sta], dct_param, dct_trigger)
                # self.picks[k][sta] = self._get_dct_picks(config, self.detections[k]['pred'][sta])
    # def get_dct_picks(dct_dets, dct_param, dct_trigger):
    def _get_initial_picks(self, config, dct_dets):
        """
        Applies optional conditions to improve phase detection, depending on which some preliminary picks are removed or kept.
        Returns dictionary containing preliminary phase picks which will be refined applying Monte Carlo Dropout MCD technique.
        ----------
        dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        """
        #
        # user-defined parameters
        #
        op_conds = config.picking['op_conds']
        dt_PS_max = config.picking['dt_PS_max'] # seconds
        tp_th_add = config.picking['tp_th_add'] # seconds
        dt_sdup_max = config.picking['dt_sdup_max'] # seconds
        dt_sp_near = config.picking['dt_sp_near'] # seconds
        #
        p_picks = [pick[0] for pick in dct_dets['p_picks']]
        s_picks = [pick[0] for pick in dct_dets['s_picks']]
        p_trigs = np.array([trig for trig in dct_dets['p_trigs'] if trig[1] != trig[0]])
        s_trigs = np.array([trig for trig in dct_dets['s_trigs'] if trig[1] != trig[0]])
        print(f"detected picks (P, S): {len(p_picks)}, {len(p_trigs)}, {len(s_picks)}, {len(s_trigs)}")
        tt_p_arg = [pick[1] for pick in dct_dets['p_picks']]
        tt_s_arg = [pick[1] for pick in dct_dets['s_picks']]
        tt = dct_dets['tt']
        prob_P = dct_dets['ts'][:,0]
        prob_S = dct_dets['ts'][:,1]
        prob_N = dct_dets['ts'][:,2]
        #
        tpicks_ml_p = np.array([t.timestamp for t in p_picks])
        tpicks_ml_s = np.array([t.timestamp for t in s_picks])
        # print(tpicks_ml_p)
        # print(tpicks_ml_s)
        #
        p_picks_bool = np.full(len(tpicks_ml_p), True)
        s_picks_bool = np.full(len(tpicks_ml_s), True)
        p_arg_selected = np.where(p_picks_bool)[0]
        s_arg_selected = np.where(s_picks_bool)[0]
        s_arg_used = []
        samp_dt = 1 / config.params['samp_freq']
        #
        # (1) Iterate over predicted P picks, in order to resolve between P and S phases predicted close in time
        #
        if '1' in op_conds:
            #
            for i, tp in enumerate(tpicks_ml_p[:]):
                #
                # search S picks detected nearby P phases
                #
                cond_pre = prob_P[:tt_p_arg[i]] > .5
                cond_pre = cond_pre[::-1]
                if len(cond_pre) > 0:
                    tp_th_pre = tp - (np.argmin(cond_pre) * config.trigger['n_shift'] * samp_dt) - tp_th_add
                else:
                    tp_th_pre = tp - tp_th_add
                #
                cond_pos = prob_P[tt_p_arg[i]:] > .5
                tp_th_pos = tp + (np.argmin(cond_pos) * config.trigger['n_shift'] * samp_dt) + tp_th_add
                #
                ts_in_th = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss >= tp_th_pre and tss <= tp_th_pos]
                #
                # picks detected before and after current P pick
                #
                tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > tp - dt_PS_max and tpp < tp_th_pre]
                tp_in_next = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp >= tp_th_pos and tpp <= tp + dt_PS_max]
                ts_in_next = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss >= tp_th_pos and tss <= tp + dt_PS_max]
                #
                if len(ts_in_th) > 0:
                    #
                    # pick = P/S or S/P
                    s_arg_used.append(ts_in_th[0][0])
                    #
                    if prob_P[tt_p_arg[i]] >= prob_S[tt_s_arg[ts_in_th[0][0]]]:
                        #
                        # P kept, S discarded
                        s_picks_bool[ts_in_th[0][0]] = False
                    else:
                        #
                        if (len(tp_in_next) > 0 or len(ts_in_next) > 0):
                            #
                            if len(tp_in_prior) == 0:
                                #
                                # next pick found, prior pick not found --> P kept, S discarded
                                s_picks_bool[ts_in_th[0][0]] = False
                            #
                            if len(tp_in_prior) > 0:
                                p_picks_bool_prior = [tpp[0] for t, tpp in enumerate(tp_in_prior) if p_picks_bool[tpp[0]]]
                                if len(p_picks_bool_prior) == 0:
                                    #
                                    # next pick found, prior pick not found --> P kept, S discarded
                                    s_picks_bool[ts_in_th[0][0]] = False
                                #
                                else:
                                    # next pick found, prior pick found --> S kept, P discarded
                                    p_picks_bool[i] = False
                        #
                        else:
                            if len(tp_in_prior) == 0:
                                #
                                # next pick not found, prior pick not found
                                # --> possibly actual S, but no prior P detected --> P and S discarded
                                p_picks_bool[i] = False
                                s_picks_bool[ts_in_th[0][0]] = False
                            #
                            else:
                                p_picks_bool_prior = [tpp[0] for t, tpp in enumerate(tp_in_prior) if p_picks_bool[tpp[0]]]
                                if len(p_picks_bool_prior) == 0:
                                    #
                                    # next pick not found, prior pick not found --> P and S discarded
                                    p_picks_bool[i] = False
                                    s_picks_bool[ts_in_th[0][0]] = False
                                #
                                else:
                                    # next pick not found, prior pick found --> S kept, P discarded
                                    p_picks_bool[i] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (2) iterate over predicted S picks, which were not handled in iteration over P picks done in (1)
        # --> S picks for which there is no earlier P or P-S predicted picks will be discarded
        #
        if '2' in op_conds:
            #
            s_arg_nonused = [i for i, ts in enumerate(tpicks_ml_s) if i not in s_arg_used]
            for i in s_arg_nonused:
                #
                ts = tpicks_ml_s[i]
                #
                # P picks detected before current S pick
                #
                tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > ts - dt_PS_max and tpp < ts and p_picks_bool[t]]
                #
                if len(tp_in_prior) == 0:
                    #
                    # prior pick not found --> discard
                    s_picks_bool[i] = False
                #
                if len(tp_in_prior) > 0:
                    #
                    tp_prior = tp_in_prior[-1][1]
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
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (3) iterate over selected S picks in order to resolve between possible duplicated S phases
        #
        if '3' in op_conds:
            #
            s_arg_used_dup = []
            dct_s_dup = {}
            for i, s_arg in enumerate(s_arg_selected):
                #
                dct_s_dup[s_arg] = [s_arg]
                ts = tpicks_ml_s[s_arg]
                cond_pos = prob_S[tt_s_arg[s_arg]:] > .5
                ts_th_pos = ts + (np.argmin(cond_pos) * config.trigger['n_shift'] * samp_dt)
                #
                for j, s_arg2 in enumerate(s_arg_selected[i+1: len(s_arg_selected)]):
                    #
                    ts2 = tpicks_ml_s[s_arg2]
                    cond_pre = prob_S[:tt_s_arg[s_arg2]] > .5
                    cond_pre = cond_pre[::-1]
                    ts2_th_pre = ts2 - (np.argmin(cond_pre) * config.trigger['n_shift'] * samp_dt)
                    #
                    if abs(ts_th_pos - ts2_th_pre) < dt_sdup_max:
                        dct_s_dup[s_arg].append(s_arg2)
                    else:
                        break
            #
            # for possible duplicated S phases, presumed false ones are discarded
            for s_arg in dct_s_dup:
                if len(dct_s_dup[s_arg]) > 1:
                    pb_s_dup = np.array([prob_S[tt_s_arg[s_arg_dup]] for s_arg_dup in dct_s_dup[s_arg]])
                    pb_s_dup_argmax = np.argmax(pb_s_dup)
                    s_arg_false = [s_arg3 for s_arg3 in dct_s_dup[s_arg] if s_arg3 != dct_s_dup[s_arg][pb_s_dup_argmax]]
                    for s_false in s_arg_false:
                        s_picks_bool[s_false] = False
                        s_arg_used_dup.append(s_false)
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (4) iterate over selected S picks in order to resolve between P and S phases predicted close in time, for special cases which may not be handled in (1)
        #
        if '4' in op_conds:
            #
            dct_sp_near = {}
            for i, s_arg in enumerate(s_arg_selected):
                #
                dct_sp_near[s_arg] = []
                ts = tpicks_ml_s[s_arg]
                #
                s_cond_pos = prob_S[tt_s_arg[s_arg]:] > .5
                ts_th_pos = ts + (np.argmin(s_cond_pos) * config.trigger['n_shift'] * samp_dt)
                #
                s_cond_pre = prob_S[:tt_s_arg[s_arg]] > .5
                s_cond_pre = s_cond_pre[::-1]
                ts_th_pre = ts - (np.argmin(s_cond_pre) * config.trigger['n_shift'] * samp_dt)
                #
                for j, p_arg in enumerate(p_arg_selected):
                    #
                    tp = tpicks_ml_p[p_arg]
                    #
                    p_cond_pos = prob_P[tt_p_arg[p_arg]:] > .5
                    tp_th_pos = tp + (np.argmin(p_cond_pos) * config.trigger['n_shift'] * samp_dt)
                    #
                    p_cond_pre = prob_P[:tt_p_arg[p_arg]] > .5
                    p_cond_pre = p_cond_pre[::-1]
                    # tp_th_pre = tp - (np.argmin(p_cond_pre) * config.trigger['n_shift'] * samp_dt)
                    if len(p_cond_pre) > 0:
                        tp_th_pre = tp - (np.argmin(p_cond_pre) * config.trigger['n_shift'] * samp_dt)
                    else:
                        tp_th_pre = tp
                    #
                    dt_sp_th = abs(ts_th_pos - tp_th_pre)
                    dt_ps_th = abs(tp_th_pos - ts_th_pre)
                    #
                    if dt_sp_th < dt_sp_near or dt_ps_th < dt_sp_near:
                        dct_sp_near[s_arg].append([p_arg, min(dt_sp_th, dt_ps_th)])
            #
            # for possible nearby P/S phases, presumed false ones are discarded
            for s_arg in dct_sp_near:
                if len(dct_sp_near[s_arg]) > 0:
                    #
                    pb_s_near = prob_S[tt_s_arg[s_arg]]
                    pb_p_near_arg = np.argmin([p_near[1] for p_near in dct_sp_near[s_arg]])
                    p_near_arg = dct_sp_near[s_arg][pb_p_near_arg][0]
                    pb_p_near = prob_P[tt_p_arg[p_near_arg]]
                    #
                    if pb_s_near >= pb_p_near:
                        p_picks_bool[p_near_arg] = False
                    else:
                        s_picks_bool[s_arg] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # print selected picks
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
        dct_picks = {
            'P': {}, 'S': {},
        }
        dct_picks['P']['pick'] = tpicks_ml_p
        dct_picks['P']['trig'] = p_trigs
        dct_picks['P']['pb'] = np.array([prob_P[tt_arg] for i, tt_arg in enumerate(tt_p_arg)])
        dct_picks['P']['bool'] = p_picks_bool
        dct_picks['P']['true_arg'] = p_arg_selected
        #
        dct_picks['S']['pick'] = tpicks_ml_s
        dct_picks['S']['trig'] = s_trigs
        dct_picks['S']['pb'] = np.array([prob_S[tt_arg] for i, tt_arg in enumerate(tt_s_arg)])
        dct_picks['S']['bool'] = s_picks_bool
        dct_picks['S']['true_arg'] = s_arg_selected
        #
        return dct_picks


                            # # dct_mcd = get_predicted_pick(best_model_pick['P'], best_params_pick['P'], dct_param, dct_trigger, data_P, sta, tpick_win, opath, ii, flag_data, save_plot)
                            # dct_mcd = self._get_predicted_pick(config, model, data_P, sta, tpick_win, opath, ii, save_plot)
    # def get_predicted_pick(best_model, best_params, dct_param, dct_trigger, data, sta, tpick_det, opath, plot_num, flag_data, save_plot=True):
    def _get_predicted_pick(self, config, model, data, sta, tpick_det, opath, plot_num, save_plot=True):
        """
        Gets refined P- or S-phase time onset and its uncertainty, by applying Monte Carlo Dropout (MCD) on the input seismic data.
        Returns a dictionary containing relevant statistics of the computed pick.
        ----------
        best_model: best (keras) model trained for phase picking.
        best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        data: (numpy array) array containing seismic stream amplitudes on which MCD is applied.
        sta: (str) station code of seismic stream.
        tpick_det: (float) preliminary phase time (in seconds, within phase seismic window) onset obtained from phase detection.
        opath: (str) output path for saving plots of predicted phase onsets.
        plot_num: (int) index of processed phase onset, used for naming plots of predicted phase onsets.
        flag_data: (str) flag defining which data is used for making predictions.
        save_plot: (bool) True to save plots of predicted phase onset.
        """
        #
        # apply Monte Carlo Dropout to get predicted time onset with uncertainty
        #
        mc_iter = config.picking['mcd_iter']
        mc_pred = []
        for j in tqdm.tqdm(range(mc_iter)):
            # x_mc = data.reshape(1, data.shape[1], data.shape[2])
            x_mc = data
            y_mc = model['best_model'].predict(x_mc, batch_size=model['best_params']['batch_size'], verbose=0)
            mc_pred.append(y_mc)
        #
        mc_pred = np.array(mc_pred)[:,0,:,:] # mc_pred.shape = (mc_iter, win_size, 1)
        mc_pred_mean = mc_pred.mean(axis=0)
        mc_pred_mean_class = (mc_pred_mean > .5).astype('int32')
        mc_pred_mean_arg_pick = mc_pred_mean_class.argmax(axis=0)[0]
        mc_pred_mean_tpick = mc_pred_mean_arg_pick / config.params['samp_freq']
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
        # this is to avoid taking into account samples where the mean class probability (mc_pred_mean) satisfies cond beyond the first intersection of
        # mc_pred_mean and prob_th1 (before the predicted time onset mc_pred_mean_tpick) or prob_th2 (after mc_pred_mean_tpick)
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
        mc_pred_mean_tpick_th1 = samps_th[0] / config.params['samp_freq']
        mc_pred_mean_tpick_th2 = samps_th[-1] / config.params['samp_freq']
        # mc_pred_mean_tres = tpick_det - mc_pred_mean_tpick
        #
        # print(tpick_det, mc_pred_mean_tpick, mc_pred_mean_tres, mc_pred_mean_tpick_th1, mc_pred_mean_tpick_th2)
        # print(tpick_det, mc_pred_mean_tpick, mc_pred_mean_tpick_th1, mc_pred_mean_tpick_th2)
        print(tpick_det, mc_pred_mean_tpick, mc_pred_mean_tpick_th1, mc_pred_mean_tpick_th2, opath, sta)
        #
        # pick class
        #
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
        dct_mcd = {
            'pick': {
                'tpick_det': tpick_det,
                'tpick': mc_pred_mean_tpick,
                'tpick_th1': mc_pred_mean_tpick_th1,
                'tpick_th2': mc_pred_mean_tpick_th2,
                'pick_class': pick_class,
                'terr_pre': terr_pre,
                'terr_pos': terr_pos,
            },
            #
            'mcd': {
                'mc_pred': mc_pred,
                'mc_pred_mean': mc_pred_mean,
                'mc_pred_mean_class': mc_pred_mean_class,
                'mc_pred_mean_arg_pick': mc_pred_mean_arg_pick,
                'mc_pred_std': mc_pred_std,
                'mc_pred_std_pick': mc_pred_std_pick,
                'prob_th1': prob_th1,
                'prob_th2': prob_th2,
            }
        }
        #
        if save_plot:
            if data.shape[2] == 1:
                # plot_predicted_phase_P(dct_mcd, dct_param, data, sta, opath, plot_num, save_plot)
                util.plot_predicted_phase_P(config, dct_mcd, data, sta, opath, plot_num, save_plot)
            elif data.shape[2] == 2:
                # plot_predicted_phase_S(dct_mcd, dct_param, data, sta, opath, plot_num, save_plot)
                util.plot_predicted_phase_S(config, dct_mcd, data, sta, opath, plot_num, save_plot)
        #
        return dct_mcd


    # def save_pick_stats(dct_picks, dct_dets, dct_data):
    def _save_pick_stats(self, dct_picks, dct_dets, dct_data):
        """
        Saves statistics of individual predicted phase onsets.
        ----------
        dct_picks: (dict) dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
        dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
        """
        #
        stas = list(dct_data['st'].keys())
        opath = dct_dets['pred'][stas[0]]['opath']
        ofile = open(f"{opath}/pick_stats/pick_stats",'a')
        util.export_dict2pckl(dct_picks, f"{opath}/pick_stats/picks.pckl")
        #
        for sta in dct_picks:
            #
            for i, k in enumerate(dct_picks[sta]['P']['true_arg']):
                #
                pick_pb = dct_picks[sta]['P']['twd'][k]['pb_win']
                tpick_det = dct_picks[sta]['P']['twd'][k]['pick_ml']['tpick_det']
                tpick_pred = dct_picks[sta]['P']['twd'][k]['pick_ml']['tpick']
                terr_pre = dct_picks[sta]['P']['twd'][k]['pick_ml']['terr_pre']
                terr_pos = dct_picks[sta]['P']['twd'][k]['pick_ml']['terr_pos']
                tpick_th1 = dct_picks[sta]['P']['twd'][k]['pick_ml']['tpick_th1']
                tpick_th1 = dct_picks[sta]['P']['twd'][k]['pick_ml']['tpick_th2']
                pick_class = dct_picks[sta]['P']['twd'][k]['pick_ml']['pick_class']
                #
                tstart_win = dct_picks[sta]['P']['twd'][k]['tstart_win']
                tpick_det_abs = tstart_win + tpick_det
                tpick_pred_abs = tstart_win + tpick_pred
                #
                pb_std = dct_picks[sta]['P']['twd'][k]['mc_ml']['mc_pred_std_pick']
                mc_pred_mean = dct_picks[sta]['P']['twd'][k]['mc_ml']['mc_pred_mean']
                mc_pred_mean_arg_pick = dct_picks[sta]['P']['twd'][k]['mc_ml']['mc_pred_mean_arg_pick']
                pb = mc_pred_mean[mc_pred_mean_arg_pick, 0]
                #
                # outstr = f"{sta} 'P' {i+1} {tpick_det_abs} {tpick_pred_abs} {tpick_det} {tpick_pred} {terr_pre} {terr_pos} {pick_class} {pb} {pb_std}"
                outstr = f"{sta} 'P' {i+1} {pick_pb:.5f} {tpick_det_abs} {tpick_pred_abs} {tpick_det:.3f} {tpick_pred:.3f} {terr_pre:.5f} {terr_pos:.5f} {pick_class} {pb:.5f} {pb_std:.5f}"
                ofile.write(outstr + '\n')
            #
            for i, k in enumerate(dct_picks[sta]['S']['true_arg']):
                #
                pick_pb = dct_picks[sta]['S']['twd'][k]['pb_win']
                tpick_det = dct_picks[sta]['S']['twd'][k]['pick_ml']['tpick_det']
                tpick_pred = dct_picks[sta]['S']['twd'][k]['pick_ml']['tpick']
                terr_pre = dct_picks[sta]['S']['twd'][k]['pick_ml']['terr_pre']
                terr_pos = dct_picks[sta]['S']['twd'][k]['pick_ml']['terr_pos']
                tpick_th1 = dct_picks[sta]['S']['twd'][k]['pick_ml']['tpick_th1']
                tpick_th1 = dct_picks[sta]['S']['twd'][k]['pick_ml']['tpick_th2']
                pick_class = dct_picks[sta]['S']['twd'][k]['pick_ml']['pick_class']
                #
                tstart_win = dct_picks[sta]['S']['twd'][k]['tstart_win']
                tpick_det_abs = tstart_win + tpick_det
                tpick_pred_abs = tstart_win + tpick_pred
                #
                pb_std = dct_picks[sta]['S']['twd'][k]['mc_ml']['mc_pred_std_pick']
                mc_pred_mean = dct_picks[sta]['S']['twd'][k]['mc_ml']['mc_pred_mean']
                mc_pred_mean_arg_pick = dct_picks[sta]['S']['twd'][k]['mc_ml']['mc_pred_mean_arg_pick']
                pb = mc_pred_mean[mc_pred_mean_arg_pick, 0]
                #
                # outstr = f"{sta} 'S' {i+1} {tpick_det_abs} {tpick_pred_abs} {tpick_det} {tpick_pred} {terr_pre} {terr_pos} {pick_class} {pb} {pb_std}"
                outstr = f"{sta} 'S' {i+1} {pick_pb:.5f} {tpick_det_abs} {tpick_pred_abs} {tpick_det:.3f} {tpick_pred:.3f} {terr_pre:.5f} {terr_pos:.5f} {pick_class} {pb:.5f} {pb_std:.5f}"
                ofile.write(outstr + '\n')
        #
        ofile.close()


    # TODO: create method: read_picks. To read saved picks dict into model object
    # def run_picking(best_params, best_model_pick, best_params_pick, dct_dets, dct_data, dct_param, dct_trigger, dct_out, save_plot=True, save_stat=True):
    def run_picking(self, config, data, model, save_plot=True, save_stat=True):
        """
        Performs P- and S-phase picking task.
        Returns dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
        ----------
        best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
        best_model_pick: (dict) dictionary containing the best models trained for phase picking.
        best_params_pick: (dict) dictionary of best performing hyperparameters optimized for phase picking.
        dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        dct_out: (dict) dictionary defining DeepPhasePick output options.
        save_plot: (bool) True to save plots of individual predicted phase onsets.
        save_stat: (bool) True to save statistics of individual predicted phase onsets.
        """
        # flag_data = dct_out['flag_data']
        # dct_picks = {}
        self.picks = {}
        best_params = model.model_detection['best_params']
        samp_dt = 1 / config.params['samp_freq']
        # for k in dct_dets:
        for k in self.detections:
            self.picks[k] = {}
            #
            tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
            ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
            tstart, tend = data.data[k]['twin']
            stas = list(data.data[k]['st'].keys())
            if len(stas) == 0:
                continue
            opath = self.detections[k]['pred'][stas[0]]['opath']
            stas_tp = []
            for sta in stas:
                if len(self.detections[k]['pred'][sta]['p_picks']) == 0:
                    continue
                stas_tp.append(sta)
            #
            print(stas_tp)
            for sta in stas_tp:
                #
                # predicted and detected picks
                print("#")
                p_picks = np.array(self.detections[k]['pred'][sta]['p_picks'])
                s_picks = np.array(self.detections[k]['pred'][sta]['s_picks'])
                # dct_picks[k][sta] = get_dct_picks(self.detections[k]['pred'][sta], dct_param, dct_trigger)
                self.picks[k][sta] = self._get_initial_picks(config, self.detections[k]['pred'][sta])
                #
                # get P-phase windows
                #
                phase = 'P'
                self.picks[k][sta][phase]['twd'] = {}
                for ii, i in enumerate(self.picks[k][sta][phase]['true_arg']):
                    #
                    self.picks[k][sta][phase]['twd'][i] = {}
                    trig = self.picks[k][sta][phase]['trig'][i]
                    y_prob = self.detections[k]['pred'][sta]['ts'][:,0]
                    x_prob = self.detections[k]['pred'][sta]['tt'] + tp_shift
                    #
                    prob_arg = np.argmax(y_prob[trig[0]:trig[1]]) + trig[0]
                    twd_1 = best_params['frac_dsamp_p1'] * best_params['win_size'] * samp_dt
                    twd_2 = best_params['win_size'] * samp_dt - twd_1
                    #
                    chs = ["N", "E", "Z"]
                    for ch in chs:
                        #
                        tr_tmp = data.data[k]['st'][sta].select(channel='*'+ch)[0]
                        #
                        tstart_win = tr_tmp.stats.starttime + x_prob[prob_arg] - twd_1
                        tend_win = tr_tmp.stats.starttime + x_prob[prob_arg] + twd_2
                        self.picks[k][sta][phase]['twd'][i]['pb_win'] = y_prob[prob_arg]
                        self.picks[k][sta][phase]['twd'][i]['tstart_win'] = tstart_win
                        self.picks[k][sta][phase]['twd'][i]['tend_win'] = tend_win
                        tpick_win = best_params['frac_dsamp_p1'] * best_params['win_size'] * samp_dt
                        self.picks[k][sta][phase]['twd'][i]['pick_ml_det'] = tpick_win
                        #
                        # waveform trace (input for CNN)
                        #
                        self.picks[k][sta][phase]['twd'][i][ch] = tr_tmp.slice(tstart_win, tend_win)
                        #
                        # correct picks: preliminary phase detection picks --> refined phase picking picks
                        #
                        # if ch == 'Z' and dct_trigger['mcd']['run_mcd']:
                        if ch == 'Z' and config.picking['run_mcd']:
                            data_P = []
                            data_P.append(self.picks[k][sta][phase]['twd'][i]['Z'].data[:-1])
                            data_P.append(self.picks[k][sta][phase]['twd'][i]['E'].data[:-1])
                            data_P.append(self.picks[k][sta][phase]['twd'][i]['N'].data[:-1])
                            data_P /= np.abs(data_P).max() # normalize before predicting picks
                            data_P = data_P[:1]
                            print("#")
                            print(f"pick: {ii+1}/{len(self.picks[k][sta][phase]['true_arg'])}")
                            # print(f"data_P: {data_P.shape}")
                            data_P = data_P.reshape(1, data_P.shape[1], 1)
                            # print(f"data_P: {data_P.shape}")
                            # print(data_P.mean(), data_P.min(), data_P.max())
                            # dct_mcd = get_predicted_pick(best_model_pick['P'], best_params_pick['P'], dct_param, dct_trigger, data_P, sta, tpick_win, opath, ii, flag_data, save_plot)
                            dct_mcd = self._get_predicted_pick(config, model.model_picking_P, data_P, sta, tpick_win, opath, ii, save_plot)
                            self.picks[k][sta][phase]['twd'][i]['pick_ml'] = dct_mcd['pick']
                            self.picks[k][sta][phase]['twd'][i]['mc_ml'] = dct_mcd['mcd']
                #
                # get S-phase windows
                #
                phase = 'S'
                self.picks[k][sta][phase]['twd'] = {}
                for ii, i in enumerate(self.picks[k][sta][phase]['true_arg']):
                    #
                    self.picks[k][sta][phase]['twd'][i] = {}
                    trig = self.picks[k][sta][phase]['trig'][i]
                    y_prob = self.detections[k]['pred'][sta]['ts'][:,1]
                    x_prob = self.detections[k]['pred'][sta]['tt'] + ts_shift
                    #
                    prob_arg = np.argmax(y_prob[trig[0]:trig[1]]) + trig[0]
                    twd_1 = best_params['frac_dsamp_s1'] * best_params['win_size'] * samp_dt
                    twd_2 = best_params['win_size'] * samp_dt - twd_1
                    #
                    for ch in chs:
                        #
                        tr_tmp = data.data[k]['st'][sta].select(channel='*'+ch)[0]
                        #
                        tstart_win = tr_tmp.stats.starttime + x_prob[prob_arg] - twd_1
                        tend_win = tr_tmp.stats.starttime + x_prob[prob_arg] + twd_2
                        self.picks[k][sta][phase]['twd'][i]['pb_win'] = y_prob[prob_arg]
                        self.picks[k][sta][phase]['twd'][i]['tstart_win'] = tstart_win
                        self.picks[k][sta][phase]['twd'][i]['tend_win'] = tend_win
                        tpick_win = best_params['frac_dsamp_s1'] * best_params['win_size'] * samp_dt
                        self.picks[k][sta][phase]['twd'][i]['pick_ml_det'] = tpick_win
                        #
                        # waveform trace (input for CNN)
                        #
                        self.picks[k][sta][phase]['twd'][i][ch] = tr_tmp.slice(tstart_win, tend_win)
                    #
                    # correct picks: preliminary phase detection picks --> refined phase picking picks
                    #
                    # if dct_trigger['mcd']['run_mcd']:
                    if config.picking['run_mcd']:
                        data_S = []
                        data_S.append(self.picks[k][sta][phase]['twd'][i]['Z'].data[:-1])
                        data_S.append(self.picks[k][sta][phase]['twd'][i]['E'].data[:-1])
                        data_S.append(self.picks[k][sta][phase]['twd'][i]['N'].data[:-1])
                        data_S = np.array(data_S)
                        data_S /= np.abs(data_S).max()
                        data_S = data_S[-2:]
                        print("#")
                        print(f"pick: {ii+1}/{len(self.picks[k][sta][phase]['true_arg'])}")
                        # print(f"data_S: {data_S.shape}")
                        data_S = data_S.T.reshape(1, data_S.shape[1], 2)
                        # print(f"data_S: {data_S.shape}")
                        # print(data_S.mean(), data_S.min(), data_S.max())
                        # dct_mcd = get_predicted_pick(best_model_pick['S'], best_params_pick['S'], dct_param, dct_trigger, data_S, sta, tpick_win, opath, ii, flag_data, save_plot)
                        dct_mcd = self._get_predicted_pick(config, model.model_picking_S, data_S, sta, tpick_win, opath, ii, save_plot)
                        self.picks[k][sta][phase]['twd'][i]['pick_ml'] = dct_mcd['pick']
                        self.picks[k][sta][phase]['twd'][i]['mc_ml'] = dct_mcd['mcd']
            #
            if save_stat:
                self._save_pick_stats(self.picks[k], self.detections[k], data.data[k])
        #
        # return dct_picks


