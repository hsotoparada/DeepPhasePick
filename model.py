# coding: utf-8

"""
This module contains a class and methods related to the phase detection and picking models integrated in DeepPhasePick method.

Author: Hugo Soto Parada (October, 2020)
Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com

"""

import util
import numpy as np
from obspy.signal.trigger import trigger_onset
from obspy.io.mseed.core import InternalMSEEDError
from tensorflow.keras.models import load_model
import tqdm
import re, sys, os, shutil, gc


class Model():
    """
    Class defining model-related methods.

    Parameters
    ----------
    version_det: str, optional
        version of optimized pre-trained model for phase detection.
    version_pick_P: str, optional
        version of optimized pre-trained model for P-phase picking.
    version_pick_S: str, optional
        version of optimized pre-trained model for S-phase picking.
    batch_size_det: int, optional
        batch size used for phase detection stage at prediction time. By default is set to the batch size optimized for the trained model.
    batch_size_pick_P: int, optional
        batch size used for P-phase picking stage at prediction time. By default is set to the batch size optimized for the trained model.
    batch_size_pick_S: int, optional
        batch size used for S-phase picking stage at prediction time. By default is set to the batch size optimized for the trained model.
    verbose: bool, optional
        If True, prints out information related to selected optimized model.
    """

    def __init__(
        self,
        version_det="20201002",
        version_pick_P="20201002_1",
        version_pick_S="20201002_1",
        batch_size_det=None,
        batch_size_pick_P=None,
        batch_size_pick_S=None,
        verbose=True,
    ):

        self.version_det = version_det
        self.version_pick_P = version_pick_P
        self.version_pick_S = version_pick_S
        self.verbose = verbose

        if self.version_det == "20201002":
            self.ntrials_det = 1000
        if self.version_pick_P in ["20201002_1", "20201002_2"]:
            self.ntrials_P = 50
        if self.version_pick_S in ["20201002_1", "20201002_2"]:
            self.ntrials_S = 50

        self.model_detection = self._get_model_detection(verbose=self.verbose)
        self.model_picking_P = self._get_model_picking(mode='P', verbose=self.verbose)
        self.model_picking_S = self._get_model_picking(mode='S', verbose=self.verbose)

        if batch_size_det is None:
            self.model_detection['batch_size_pred'] = self.model_detection['best_params']['batch_size']
        else:
            self.model_detection['batch_size_pred'] = batch_size_det

        if batch_size_pick_P is None:
            self.model_picking_P['batch_size_pred'] = self.model_picking_P['best_params']['batch_size']
        else:
            self.model_picking_P['batch_size_pred'] = batch_size_pick_P

        if batch_size_pick_S is None:
            self.model_picking_S['batch_size_pred'] = self.model_picking_S['best_params']['batch_size']
        else:
            self.model_picking_S['batch_size_pred'] = batch_size_pick_S


    def _get_model_detection(self, verbose=True):
        """
        Read best model and relevant results obtained from the hyperparameter optimization for phase detection.

        Parameters
        ----------
        verbose: bool, optional
            If True, prints out information related to selected optimized model.

        Returns
        -------
        dct: dict
            Dictionary containing best model and relevant related results.
        """
        #
        ipath = f"models/detection/{self.version_det}"
        trials = util.import_pckl2dict(f"{ipath}/trials_hyperopt_ntrials_{self.ntrials_det:03}.pckl")
        arg_best_trial = util.get_arg_best_trial(trials)
        best_results = util.import_pckl2dict(f"{ipath}/dict_hyperopt_t{arg_best_trial+1:03}.pckl")
        best_params = best_results['params']
        best_model = load_model(f"{ipath}/model_hyperopt_t{arg_best_trial+1:03}.h5")
        best_hist = best_results['history']
        #
        if verbose:
            print("#")
            print(best_model.summary())
            #
            print("######")
            print(f"best model for phase detection found for trial {arg_best_trial+1:03}/{self.ntrials_det:03} and hyperparameters:")
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


    def _get_model_picking(self, mode, verbose=True):
        """
        Read best model and relevant results obtained from hyperparameter optimization for phase picking.

        Parameters
        ----------
        mode: str
            'P' or 'S' for retrieving P- or S- phase picking model, respectively.
        verbose: bool, optional
            If True, prints out information related to selected optimized model.

        Returns
        -------
        dct: dict
            Dictionary containing best model and relevant related results.
        """
        #
        if mode == 'P':
            ipath = f"models/picking/{self.version_pick_P}/P"
            ntrials = self.ntrials_P
        else:
            ipath = f"models/picking/{self.version_pick_S}/S"
            ntrials = self.ntrials_S
        #
        trials = util.import_pckl2dict(f"{ipath}/trials_hyperopt_ntrials_{ntrials:03}.pckl")
        arg_best_trial = util.get_arg_best_trial(trials)
        best_results = util.import_pckl2dict(f"{ipath}/dict_hyperopt_t{arg_best_trial+1:03}.pckl")
        best_params = best_results['params']
        best_model = load_model(f"{ipath}/model_hyperopt_t{arg_best_trial+1:03}.h5")
        best_hist = best_results['history']
        #
        if verbose:
            print("#")
            print(best_model.summary())
            #
            print("######")
            print(f"best model for {mode} phase picking found for trial {arg_best_trial+1:03}/{ntrials:03} and hyperparameters:")
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


    def _sliding_window(self, data, size, stepsize=1, axis=-1):
        """
        Calculates a sliding window over data.
        Adapted from similar function in https://github.com/interseismic/generalized-phase-detection (see Ross et al., 2018; doi:10.1785/0120180080)

        Parameters
        ----------
        data: ndarray
            1D array containing data to be slided over.
        size: int
            sliding window size.
        stepsize: int
            sliding window stepsize.
        axis: int
            axis to slide over. Defaults to the last axis.

        Returns
        -------
        dct: ndarray
            2D array where rows represent the sliding windows.

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

        # print('sliding window', data.shape, data.ndim)
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
        # print('sliding window - strided', strided.shape, strided.ndim)
        return strided.copy()


    def _make_prediction(self, config, st):
        """
        Applies best pre-trained phase detection model on waveform data to calculate P- and S-phase discrete probability time series from predictions.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
        st: instance of obspy.core.Stream
            Seismic stream on which predictions are made.

        Returns
        -------
        Tuple containing predicted P- and S-phase discrete probability time series computed from predictions.
        """
        #
        # trim traces within common start and end times to avoid error:
        # --> ValueError: could not broadcast input array from shape (17958,460) into shape (17963,460)
        tstart_arr = np.array([tr.stats.starttime for tr in st])
        tend_arr = np.array([tr.stats.endtime for tr in st])
        tstart_cond = (tstart_arr == st[0].stats.starttime)
        tend_cond = (tend_arr == st[0].stats.endtime)
        st_trim_flag = False
        #
        if tstart_cond.sum() != len(tstart_arr) or tend_cond.sum() != len(tend_arr):
            # print(f"strimming stream: {tstart_cond.sum()}, {tend_cond.sum()}")
            print(f"strimming stream...")
            st_trim_flag = True
            st.trim(tstart_arr.max(), tend_arr.min())
        #
        # Preparing data matrix for sliding window
        # print("Preparing data matrix for sliding window")
        #
        st_data = [st[0].data, st[1].data, st[2].data]
        #
        best_params = self.model_detection['best_params']
        best_model = self.model_detection['best_model']
        dt = st[0].stats.delta
        # print(st_data[0].size, config.trigger['n_shift'], best_params['win_size'], dt)
        tt = (np.arange(0, st_data[0].size, config.trigger['n_shift']) + .5 * best_params['win_size']) * dt #[sec]
        # print(tt.shape, tt.ndim)
        #
        try:
            sliding_E = self._sliding_window(st_data[0], best_params['win_size'], stepsize=config.trigger['n_shift'])
            sliding_N = self._sliding_window(st_data[1], best_params['win_size'], stepsize=config.trigger['n_shift'])
            sliding_Z = self._sliding_window(st_data[2], best_params['win_size'], stepsize=config.trigger['n_shift'])
            tr_win = np.zeros((sliding_N.shape[0], best_params['win_size'], 3))
            tr_win[:,:,0] = sliding_E
            tr_win[:,:,1] = sliding_N
            tr_win[:,:,2] = sliding_Z
            #
            # normalization, in separated operations to avoid memory errors
            aa = np.abs(tr_win)
            bb = np.max(aa, axis=(1,2))[:,None,None]
            tr_win = tr_win / bb
            #tr_win = tr_win / np.max(np.abs(tr_win), axis=(1,2))[:,None,None]
            #
            tt = tt[:tr_win.shape[0]]
            # print(tt.shape, tt.ndim)
            #
            # make model predictions
            ts = best_model.predict(tr_win, verbose=True, batch_size=self.model_detection['batch_size_pred'])
            #
            prob_P = ts[:,0]
            prob_S = ts[:,1]
            prob_N = ts[:,2]
        except ValueError:
            tt, ts, prob_S, prob_P, prob_N = [0],[0],[0],[0],[0]
        #
        return (tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st)


    def _calculate_trigger(self, config, st, net, sta, tt, ts, prob_P, prob_S):
        """
        Calculates trigger on and off times of P- and S-phases, from predicted discrete P,S-class probability time series.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
        st: instance of obspy.core.Stream
            Seismic stream on which predictions were made and trigger time are calculated.
        net: str
            Network code of seismic stream.
        sta: str
            Station code of seismic stream.
        tt: ndarray
            1D array of times within seismic stream at which discrete probabilities are assigned (center of sliding windows).
        ts: ndarray
            2D array containing discrete probability time series of predicted P, S, and N classes.
        prob_P: ndarray
            1D array containing discrete probability time series of predicted P class.
        prob_S: ndarray
            1D array containing discrete probability time series of predicted S class.

        Returns
        -------
        Tuple containing preliminary P- and S-phase onsets and trigger on/off times.
        """
        #
        #trigger_onset(charfct, thres1, thres2, max_len=9e+99, max_len_delete=False)
        #calculates trigger on and off times from characteristic function charfct, given thresholds thres1 and thres2
        #
        #correction of time position for P, S predicted picks
        #
        best_params = self.model_detection['best_params']
        samp_dt = 1 / config.data_params['samp_freq']
        tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
        ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
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
        return (p_picks, s_picks, p_trigs, s_trigs)
        ofile.close()


    def run_detection(self, config, data, save_dets=False, save_data=False):
        """
        Performs P- and S-phase detection.
        Computes discrete class probability time series from predictions, which are used to obtain preliminary phase picks.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
        data: instance of data.Data
            Contains selected seismic waveform data on which phase detection is applied.
        save_dets: bool
            If True, saves a dictionary containing predicted discrete class probability time series and preliminary phase picks.
        save_data: bool
            If True, saves a dictionary containing seismic waveform data on which phase detection is applied.
        """
        #
        # detect seismic phases on processed stream data
        #
        self.detections = {}
        for i in data.data:
            self.detections[i] = {}
            opath = data.data[i]['opath']
            for s in sorted(data.data[i]['st'].keys())[:]:
                #
                st_tmp = data.data[i]['st'][s]
                net = st_tmp[0].stats.network
                ch = st_tmp[0].stats.channel
                dt = st_tmp[0].stats.delta
                print("#")
                print(f"Calculating predictions for stream: {net}.{s}..{ch[:-1]}?...")
                # print(st_tmp)
                #
                # get predicted discrete phase class probability time series
                #
                tr_win, tt, ts, prob_S, prob_P, prob_N, st_trim_flag, st_trimmed = self._make_prediction(config, st_tmp)
                # print(tt.shape, ts.shape, prob_S.shape, prob_P.shape, prob_N.shape)
                # print(tt.ndim, ts.ndim, prob_S.ndim, prob_P.ndim, prob_N.ndim)
                print(st_trimmed)
                #
                # skip streams raising ValueError in make_prediction()
                if len(ts) == 1 and ts[0] == 0:
                    val_err = f"Sliding window size may not exceed size of selected axis"
                    print(f"skipping stream {net}.{s}..{ch[:-1]}? due to ValueError: {val_err}...")
                    continue
                #
                # get preliminary phase picks
                #
                p_picks, s_picks, p_trigs, s_trigs = self._calculate_trigger(config, st_tmp, net, s, tt, ts, prob_P, prob_S)
                print(f"p_picks = {len(p_picks)}, s_picks = {len(s_picks)}")
                #
                self.detections[i][s] = {
                    'dt': dt, 'tt': tt, 'ts': ts,
                    'p_picks': p_picks, 's_picks': s_picks,
                    'p_trigs': p_trigs, 's_trigs': s_trigs,
                    'opath': opath,
                }
                if st_trim_flag:
                    data.data[i]['st'][s] = st_trimmed
            #
            if save_dets:
                os.makedirs(f"{opath}/pick_stats", exist_ok=True)
                util.export_dict2pckl(self.detections[i], f"{opath}/pick_stats/detections.pckl")
            if save_data:
                os.makedirs(f"{opath}/pick_stats", exist_ok=True)
                util.export_dict2pckl(data.data[i], f"{opath}/pick_stats/data.pckl")


    def read_detections(self, data):
        """
        Read predicted discrete phase class probability time series and preliminary phase picks from dictionary stored by using save_dets=True in method model.run_detection().

        Parameters
        ----------
        data: instance of data.Data
            Contains selected seismic waveform data on which phase detection has been applied.
        """
        self.detections = {}
        for i in data.data:
            ipath = data.data[i]['opath']
            self.detections[i] = util.import_pckl2dict(f"{ipath}/pick_stats/detections.pckl")


    def _get_initial_picks(self, config, dct_dets):
        """
        Applies optional conditions to improve phase detection. Some preliminary picks are removed or kept depending on these conditions.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
        dct_dets: dict
            dictionary containing predicted discrete phase class probability time series and preliminary phase picks.

        Returns
        -------
        dct_picks: dict
            Dictionary containing preliminary phase picks which to be refined by applying Monte Carlo Dropout MCD technique.
        """
        #
        # user-defined parameters
        #
        op_conds = config.picking['op_conds']
        tp_th_add = config.picking['tp_th_add'] # seconds
        dt_sp_near = config.picking['dt_sp_near'] # seconds
        dt_ps_max = config.picking['dt_ps_max'] # seconds
        dt_sdup_max = config.picking['dt_sdup_max'] # seconds
        #
        p_picks = [pick[0] for pick in dct_dets['p_picks']]
        s_picks = [pick[0] for pick in dct_dets['s_picks']]
        p_trigs = np.array([trig for trig in dct_dets['p_trigs'] if trig[1] != trig[0]])
        s_trigs = np.array([trig for trig in dct_dets['s_trigs'] if trig[1] != trig[0]])
        # print(f"triggered picks (P, S): {len(p_picks)}, {len(p_trigs)}, {len(s_picks)}, {len(s_trigs)}")
        print(f"triggered picks (P, S): {len(p_picks)}, {len(s_picks)}")
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
        samp_dt = 1 / config.data_params['samp_freq']
        #
        # (1) Iterate over predicted P picks, in order to resolve between P and S phases predicted close in time, with overlapping probability time series
        #
        if '1' in op_conds:
            #
            for i, tp in enumerate(tpicks_ml_p[:]):
                # print('1', i, tp)
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
                # tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > tp - dt_ps_max and tpp < tp_th_pre]
                # tp_in_next = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp >= tp_th_pos and tpp <= tp + dt_ps_max]
                # ts_in_next = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss >= tp_th_pos and tss <= tp + dt_ps_max]
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
                        # S kept, P discarded
                        p_picks_bool[i] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (2) iterate over selected S picks in order to resolve between P and S phases predicted close in time, with non-overlapping probability time series
        #
        if '2' in op_conds:
            #
            dct_sp_near = {}
            for i, s_arg in enumerate(s_arg_selected):
                # print('2', i, s_arg)
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
        # (3) iterate over selected S picks. S picks for which there is no earlier P or P-S predicted picks will be discarded
        #
        if '3' in op_conds:
            #
            # s_arg_nonused = [i for i, ts in enumerate(tpicks_ml_s) if i not in s_arg_used]
            # for i in s_arg_nonused:
            for i, s_arg in enumerate(s_arg_selected):
                # print('3', i)
                #
                # ts = tpicks_ml_s[i]
                ts = tpicks_ml_s[s_arg]
                #
                # P picks detected before current S pick
                #
                tp_in_prior = [(t, tpp) for t, tpp in enumerate(tpicks_ml_p) if tpp > ts - dt_ps_max and tpp < ts and p_picks_bool[t]]
                #
                if len(tp_in_prior) == 0:
                    #
                    # prior pick not found --> discard
                    # s_picks_bool[i] = False
                    s_picks_bool[s_arg] = False
                #
                if len(tp_in_prior) > 0:
                    #
                    tp_prior = tp_in_prior[-1][1]
                    ts_in_prior = [(t, tss) for t, tss in enumerate(tpicks_ml_s) if tss > tp_prior and tss < ts and t in np.where(s_picks_bool)[0]]
                    #
                    if len(ts_in_prior) > 1:
                        # s_picks_bool[i] = False
                        s_picks_bool[s_arg] = False
                    #
                    # if len(ts_in_prior) == 1:
                    #     #
                    #     ts_prior = ts_in_prior[0][1]
                    #     if ts > ts_prior + abs(tp_prior - ts_prior):
                    #         s_picks_bool[i] = False
            #
            p_arg_selected = np.where(p_picks_bool)[0]
            s_arg_selected = np.where(s_picks_bool)[0]
        #
        # (4) iterate over selected S picks in order to resolve between possible duplicated S phases
        #
        if '4' in op_conds:
            #
            s_arg_used_dup = []
            dct_s_dup = {}
            for i, s_arg in enumerate(s_arg_selected):
                # print('4', i, s_arg)
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
        # print selected picks
        #
        print(f"selected picks (P, S): {len(np.where(p_picks_bool)[0])}, {len(np.where(s_picks_bool)[0])}")
        # print("#")
        # for i, tp in enumerate(tpicks_ml_p):
        #     print(i+1, tp, p_picks_bool[i])
        #
        # print("#")
        # for i, ts in enumerate(tpicks_ml_s):
        #     print(i+1, ts, s_picks_bool[i])
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


    def _get_predicted_picks(self, config, model, data, sta, tpick_det, opath):
        """
        Gets refined P- or S-phase onset time and uncertainty by applying Monte Carlo Dropout (MCD) on the input seismic data.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
        model: dict
            dictionary containing optimized pre-trained model for P- or S-phase picking.
        data: ndarray
            3D array containing seismic stream amplitudes on which MCD is applied.
        sta: str
            Station code of seismic stream.
        tpick_det: float
            Preliminary phase time onset (in seconds, within picking window) obtained from phase detection.
        opath: str
            Output path for saving figure of predicted phase onsets.

        Returns
        -------
        dct_mcd: dict
            Dictionary containing relevant statistics of the computed pick.
        """
        #
        # apply Monte Carlo Dropout to get predicted time onset with uncertainty
        #
        # print('get_predicted_picks', data.shape, data.ndim)
        mc_iter = config.picking['mcd_iter']
        mc_pred = []
        for j in tqdm.tqdm(range(mc_iter)):
            # x_mc = data.reshape(1, data.shape[1], data.shape[2])
            x_mc = data
            y_mc = model['best_model'].predict(x_mc, batch_size=model['batch_size_pred'], verbose=0)
            mc_pred.append(y_mc)
        #
        mc_pred = np.array(mc_pred)[:,0,:,:] # mc_pred.shape = (mc_iter, win_size, 1)
        mc_pred_mean = mc_pred.mean(axis=0)
        mc_pred_mean_class = (mc_pred_mean > .5).astype('int32')
        mc_pred_mean_arg_pick = mc_pred_mean_class.argmax(axis=0)[0]
        mc_pred_mean_tpick = mc_pred_mean_arg_pick / config.data_params['samp_freq']
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
        # this restricts the uncertainty calculation to the time interval between the predicted time onset (mc_pred_mean_tpick) and the first intersections
        # (in the rare case that these are not unique) of the mean probability (mc_pred_mean) with prob_th1 (before the onset) and with prob_th2 (after the onset)
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
        mc_pred_mean_tpick_th1 = samps_th[0] / config.data_params['samp_freq']
        mc_pred_mean_tpick_th2 = samps_th[-1] / config.data_params['samp_freq']
        # mc_pred_mean_tres = tpick_det - mc_pred_mean_tpick
        #
        # print(tpick_det, mc_pred_mean_tpick, mc_pred_mean_tres, mc_pred_mean_tpick_th1, mc_pred_mean_tpick_th2)
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
        return dct_mcd


    def _save_pick_stats(self, config, dct_picks, dct_dets, dct_data):
        """
        Saves statistics of refined phase onsets.

        Parameters
        ----------
        dct_picks: dict
            Dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
        dct_dets: dict
            Dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
        dct_data: dict
            Dictionary containing seismic data on which DeepPhasePick is applied.
        """
        #
        stas = list(dct_data['st'].keys())
        opath = dct_dets[stas[0]]['opath']
        os.makedirs(f"{opath}/pick_stats", exist_ok=True)
        ofile = open(f"{opath}/pick_stats/pick_stats",'w')
        #
        for sta in dct_picks:
            #
            for i, k in enumerate(dct_picks[sta]['P']['true_arg']):
                #
                if config.picking['run_mcd']:
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
                    outstr = f"{sta} P {i+1} {pick_pb:.5f} {tpick_det_abs} {tpick_pred_abs} {tpick_det:.3f} {tpick_pred:.3f} {terr_pre:.5f} {terr_pos:.5f} {pick_class} {pb:.5f} {pb_std:.5f}"
                    ofile.write(outstr + '\n')
                else:
                    pick_pb = dct_picks[sta]['P']['twd'][k]['pb_win']
                    tpick_det = dct_picks[sta]['P']['twd'][k]['pick_ml_det']
                    tstart_win = dct_picks[sta]['P']['twd'][k]['tstart_win']
                    tpick_det_abs = tstart_win + tpick_det
                    #
                    outstr = f"{sta} P {i+1} {pick_pb:.5f} {tpick_det_abs}"
                    ofile.write(outstr + '\n')
            #
            for i, k in enumerate(dct_picks[sta]['S']['true_arg']):
                #
                if config.picking['run_mcd']:
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
                    outstr = f"{sta} S {i+1} {pick_pb:.5f} {tpick_det_abs} {tpick_pred_abs} {tpick_det:.3f} {tpick_pred:.3f} {terr_pre:.5f} {terr_pos:.5f} {pick_class} {pb:.5f} {pb_std:.5f}"
                    ofile.write(outstr + '\n')
                else:
                    pick_pb = dct_picks[sta]['S']['twd'][k]['pb_win']
                    # tpick_det = dct_picks[sta]['S']['twd'][k]['pick_ml']['tpick_det']
                    tpick_det = dct_picks[sta]['S']['twd'][k]['pick_ml_det']
                    tstart_win = dct_picks[sta]['S']['twd'][k]['tstart_win']
                    tpick_det_abs = tstart_win + tpick_det
                    #
                    outstr = f"{sta} S {i+1} {pick_pb:.5f} {tpick_det_abs}"
                    ofile.write(outstr + '\n')
        #
        ofile.close()

    def run_picking(self, config, data, save_plots=True, save_stats=True, save_picks=False):
        """
        Performs P- and S-phase picking tasks, by refining (through Monte Carlo Dropout MCD) preliminary phase picks obtained from phase detection task.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
        data: instance of data.Data
            Contains selected seismic waveform data on which phase detection is applied.
        save_plots: bool
            If True, saves figures of predicted phase onsets. Only if additionally config.picking['run_mcd'] = True.
        save_stats: bool
            If True, saves statistics of predicted phase onsets.
        save_picks: bool
            If True, saves a dictionary containing preliminary (from detection stage) and refined (from picking stage) phase picks.
        """

        self.picks = {}
        best_params = self.model_detection['best_params']
        samp_dt = 1 / config.data_params['samp_freq']
        for k in self.detections:
            self.picks[k] = {}
            #
            tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
            ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
            tstart, tend = data.data[k]['twin']
            stas = list(data.data[k]['st'].keys())
            if len(stas) == 0:
                continue
            opath = self.detections[k][stas[0]]['opath']
            stas_tp = []
            for sta in stas:
                if len(self.detections[k][sta]['p_picks']) == 0:
                    continue
                stas_tp.append(sta)
            #
            # print(stas_tp)
            for sta in stas_tp:
                #
                # predicted and detected picks
                print("#")
                print(f"{k}, {tstart}, {tend}, {sta}")
                p_picks = np.array(self.detections[k][sta]['p_picks'])
                s_picks = np.array(self.detections[k][sta]['s_picks'])
                self.picks[k][sta] = self._get_initial_picks(config, self.detections[k][sta])
                #
                # get P-phase windows
                #
                phase = 'P'
                self.picks[k][sta][phase]['twd'] = {}
                for ii, i in enumerate(self.picks[k][sta][phase]['true_arg']):
                    #
                    self.picks[k][sta][phase]['twd'][i] = {}
                    trig = self.picks[k][sta][phase]['trig'][i]
                    y_prob = self.detections[k][sta]['ts'][:,0]
                    x_prob = self.detections[k][sta]['tt'] + tp_shift
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
                        # waveform trace (input for RNN)
                        #
                        self.picks[k][sta][phase]['twd'][i][ch] = tr_tmp.slice(tstart_win, tend_win)
                        #
                        # correct picks: preliminary phase detection picks --> refined phase picking picks
                        #
                        if ch == 'Z' and config.picking['run_mcd']:
                            data_P = []
                            data_P.append(self.picks[k][sta][phase]['twd'][i]['Z'].data[:-1])
                            data_P.append(self.picks[k][sta][phase]['twd'][i]['E'].data[:-1])
                            data_P.append(self.picks[k][sta][phase]['twd'][i]['N'].data[:-1])
                            data_P /= np.abs(data_P).max() # normalize before predicting picks
                            data_P = data_P[:1]
                            print("#")
                            print(f"P pick: {ii+1}/{len(self.picks[k][sta][phase]['true_arg'])}")
                            # print(f"data_P: {data_P.shape}")
                            data_P = data_P.reshape(1, data_P.shape[1], 1)
                            # print(f"data_P: {data_P.shape}")
                            # print(data_P.mean(), data_P.min(), data_P.max())
                            dct_mcd = self._get_predicted_picks(config, self.model_picking_P, data_P, sta, tpick_win, opath)
                            self.picks[k][sta][phase]['twd'][i]['pick_ml'] = dct_mcd['pick']
                            self.picks[k][sta][phase]['twd'][i]['mc_ml'] = dct_mcd['mcd']
                            if save_plots:
                                util.plot_predicted_phase_P(config, dct_mcd, data_P, sta, opath, ii)
                #
                # get S-phase windows
                #
                phase = 'S'
                self.picks[k][sta][phase]['twd'] = {}
                for ii, i in enumerate(self.picks[k][sta][phase]['true_arg']):
                    #
                    self.picks[k][sta][phase]['twd'][i] = {}
                    trig = self.picks[k][sta][phase]['trig'][i]
                    y_prob = self.detections[k][sta]['ts'][:,1]
                    x_prob = self.detections[k][sta]['tt'] + ts_shift
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
                        # waveform trace (input for RNN)
                        #
                        self.picks[k][sta][phase]['twd'][i][ch] = tr_tmp.slice(tstart_win, tend_win)
                    #
                    # correct picks: preliminary phase detection picks --> refined phase picking picks
                    #
                    if config.picking['run_mcd']:
                        data_S = []
                        data_S.append(self.picks[k][sta][phase]['twd'][i]['Z'].data[:-1])
                        data_S.append(self.picks[k][sta][phase]['twd'][i]['E'].data[:-1])
                        data_S.append(self.picks[k][sta][phase]['twd'][i]['N'].data[:-1])
                        data_S = np.array(data_S)
                        data_S /= np.abs(data_S).max()
                        data_S = data_S[-2:]
                        print("#")
                        print(f"S pick: {ii+1}/{len(self.picks[k][sta][phase]['true_arg'])}")
                        # print(f"data_S: {data_S.shape}")
                        data_S = data_S.T.reshape(1, data_S.shape[1], 2)
                        # print(f"data_S: {data_S.shape}")
                        # print(data_S.mean(), data_S.min(), data_S.max())
                        dct_mcd = self._get_predicted_picks(config, self.model_picking_S, data_S, sta, tpick_win, opath)
                        self.picks[k][sta][phase]['twd'][i]['pick_ml'] = dct_mcd['pick']
                        self.picks[k][sta][phase]['twd'][i]['mc_ml'] = dct_mcd['mcd']
                        if save_plots:
                            util.plot_predicted_phase_S(config, dct_mcd, data_S, sta, opath, ii)
            #
            if save_stats:
                self._save_pick_stats(config, self.picks[k], self.detections[k], data.data[k])
            #
            if save_picks:
                os.makedirs(f"{opath}/pick_stats", exist_ok=True)
                util.export_dict2pckl(self.picks[k], f"{opath}/pick_stats/picks.pckl")


    def read_picks(self, data):
        """
        Read picks from dictionary, which has been previously stored by using save_picks=True in method model.run_picking().

        Parameters
        ----------
        data: instance of data.Data
            Contains selected seismic waveform data on which phase detection has been applied.
        """
        self.picks = {}
        for i in data.data:
            ipath = data.data[i]['opath']
            self.picks[i] = util.import_pckl2dict(f"{ipath}/pick_stats/picks.pckl")
