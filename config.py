# coding: utf-8

"""
This module contains a class and methods that help to configure the behavior of DeepPhasePick method.

Author: Hugo Soto Parada (October, 2020)
Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com

"""

import obspy.core as oc
from datetime import datetime
import re, sys, os, shutil, gc


class Config():
    """
    Class that initiates user configuration for selecting seismic data and defining how this data is processed in DeepPhasePick.

    Parameters
    ----------
    dct_data: dict, optional
        dictionary with parameters defining archived waveform data on which DeepPhasePick is applied.
        See parameters details in method set_data().
    dct_data_params: dict, optional
        dictionary with parameters defining how seismic waveforms is processed before phase detection.
        See parameters details in method set_data_params().
    dct_time: dict, optional
        dictionary with parameters defining time windows over which DeepPhasePick is applied.
        See parameters details in method set_time().
    dct_trigger: dict, optional
        dictionary with parameters defining how predicted discrete probability time series are computed when running phase detection on seismic waveforms.
        See parameters details in method set_trigger().
    dct_picking: dict, optional
        dictionary with parameters applied in optional conditions for improving preliminary picks obtained from phase detection.
        See parameters details in method set_picking().
    """

    def __init__(self, dct_data=None, dct_data_params=None, dct_time=None, dct_trigger=None, dct_picking=None):

        self.data = self._set_default_data(dct_data)
        self.data_params = self._set_default_data_params(dct_data_params)
        self.time = self._set_default_time(dct_time)
        self.trigger = self._set_default_trigger(dct_trigger)
        self.picking = self._set_default_picking(dct_picking)


    def _set_default_data(self, dct_data):
        """
        Set default parameters defining archived waveform data on which DeepPhasePick is applied.

        Returns
        -------
        dct: dict
            dictionary with defined parameters. See parameters details in method set_data().
        """

        # TODO: define for sample data
        dct = {
            'stas': [],
            'ch': 'HH',
            'net': '',
            'archive': '',
            'opath': 'out',
            # #
            # # 'stas': ['AB03', 'AB05', 'AB10', 'AB12', 'AB17', 'AB21', 'AB22', 'AB24', 'AB25', 'AB27'],
            # 'stas': ['AB10', 'AB21', 'AB25'],
            # # 'stas': ['AB10', 'AB21'],
            # 'ch': 'HH',
            # 'net': '9K',
            # # 'archive': 'archive',
            # 'archive': '/home/soto/Volumes/CHILE/soto_work/HART_ALBANIA/archive',
        }

        if dct_data is not None:
            for k in dct:
                if k in dct_data:
                    dct[k] = dct_data[k]

        return dct


    def _set_default_data_params(self, dct_data_params):
        """
        Set default parameters defining how seismic waveforms is processed before phase detection.

        Returns
        -------
        dct: dict
            dictionary with defined parameters. See parameters details in method set_data_params().
        """

        dct = {
            'samp_freq': 100.,
            'st_normalized': True,
            'st_detrend': True,
            'st_filter': False,
            'filter_type': 'bandpass',
            'filter_fmin': .2,
            'filter_fmax': 10.,
        }

        if dct_data_params is not None:
            for k in dct:
                if k in dct_data_params:
                    dct[k] = dct_data_params[k]

        return dct


    def _set_default_time(self, dct_time):
        """
        Set parameters defining time windows over which DeepPhasePick is applied.

        Returns
        -------
        dct: dict
            dictionary with defined parameters. See parameters details in method set_time().
        """

        # TODO: define for sample data
        dct = {
            #
            'dt_iter': 3600. * 1,
            'tstart': oc.UTCDateTime(2020, 1, 11, 21, 0, 0),
            'tend': oc.UTCDateTime(2020, 1, 11, 22, 0, 0),
            # 'tstarts': [
            #     oc.UTCDateTime(2020, 2, 16, 12, 0, 0)
            #     ],
            # 'tends': [
            #     oc.UTCDateTime(2020, 2, 17, 0, 0, 0)
            #     ],
            #
            # 'STEAD': {
            #     'dt_iter': 3600. * 1,
            #     'tstarts': [0],
            #     'tends': [0],
            # },
            # #
            # 'test_dpp_detection_1': {
            #     'dt_iter': 3600. * 1,
            #     'tstarts': [0],
            #     'tends': [0],
            # },
            # 'test_dpp_picking_1': {
            #     'dt_iter': 3600. * 1,
            #     'tstarts': [0],
            #     'tends': [0],
            # },
        }

        if dct_time is not None:
            for k in dct:
                if k in dct_time:
                    if k in ['tstart', 'tend']:
                        dct[k] = oc.UTCDateTime(dct_time[k])
                    else:
                        dct[k] = dct_time[k]

        return dct


    def _set_default_trigger(self, dct_trigger):
        """
        Set default parameters defining how predicted discrete probability time series are computed when running phase detection on seismic waveforms.

        Returns
        -------
        dct: dict
            dictionary with defined parameters. See parameters details in method set_trigger().
        """

        dct = {
            # 'n_shift': 10, 'pthres_p': [0.5, .001], 'pthres_s': [0.5, .001], 'max_trig_len': [9e99, 9e99],
            'n_shift': 10, 'pthres_p': [0.9, .001], 'pthres_s': [0.9, .001], 'max_trig_len': [9e99, 9e99],
            # 'n_shift': 10, 'pthres_p': [0.95, .001], 'pthres_s': [0.95, .001], 'max_trig_len': [9e99, 9e99],
            # 'n_shift': 10, 'pthres_p': [0.98, .001], 'pthres_s': [0.95, .001], 'max_trig_len': [9e99, 9e99],
            # 'n_shift': 10, 'pthres_p': [0.98, .001], 'pthres_s': [0.98, .001], 'max_trig_len': [9e99, 9e99],
        }

        if dct_trigger is not None:
            for k in dct:
                if k in dct_trigger:
                    dct[k] = dct_trigger[k]

        return dct


    def _set_default_picking(self, dct_picking):
        """
        Set default parameters applied in optional conditions for improving preliminary picks obtained from phase detection.

        Returns
        -------
        dct: dict
            dictionary with defined parameters. See parameters details in method set_trigger().
        """

        dct = {
            # 'op_conds': ['1', '2', '3', '4'],
            'op_conds': [],
            #
            'tp_th_add': 1.5,
            'dt_sp_near': 3.,
            'dt_ps_max': 35.,
            'dt_sdup_max': 3.,
            #
            # 'tp_th_add': 1.5,
            # 'dt_sp_near': 1.5,
            # 'dt_ps_max': 25.,
            # 'dt_sdup_max': 2.,
            #
            'run_mcd': True,
            # 'mcd_iter': 5,
            'mcd_iter': 10,
        }

        if dct_picking is not None:
            for k in dct:
                if k in dct_picking:
                    dct[k] = dct_picking[k]

        return dct


    def set_data(self, stas, ch, net, archive, opath='out'):
        """
        Set parameters defining archived waveform data on which DeepPhasePick is applied.

        Parameters
        ----------
        stas: list of str
            stations from which waveform data are used.
        ch: str
            channel code of selected waveforms.
        net: str
            network code of selected stations.
        archive: str
            path to the structured or unstructured archive where waveforms are read from.
        opath: str, optional
            output path where results are stored.
        """
        self.data = {
            'stas': stas,
            'ch': ch,
            'net': net,
            'archive': archive,
            'opath': opath,
        }


    # TODO: remove st_normalize, maybe it's better to normalize the data by default, as models were trained with normalized data
    # TODO: maybe define st_filter as an instance of obspy.filter ?? To make it more general and flexible
    def set_data_params(self, samp_freq=100., st_normalized=True, st_detrend=True, st_filter=False, filter_type='bandpass', filter_fmin=.2, filter_fmax=10.):
        """
        Set parameters defining how seismic waveforms is processed before phase detection.

        Parameters
        ----------
        samp_freq: float, optional
            sampling rate [Hz] of the seismic waveforms. Waveforms with different sampling rate should be resampled to samp_freq.
        st_normalized: bool, optional
            If True, normalize waveforms on which phase detection is performed.
        st_detrend: bool, optional
            If True, detrend (linear) waveforms on which phase detection is performed.
        st_filter: bool, optional
            If True, filter waveforms on which phase detection is performed.
        filter_type: str, optional
            type of filter applied to waveforms on which phase detection is performed.
            See obspy.core.stream.Stream.filter.
        filter_fmin: float, optional
            Minimum frequency of applied filter. Data below this frequency is removed.
        filter_fmax: float, optional
            Maximum frequency of applied filter. Data above this frequency is removed.

        """

        self.data_params = {
            'samp_freq': samp_freq,
            'st_normalized': st_normalized,
            'st_detrend': st_detrend,
            'st_filter': st_filter,
            'filter_type': filter_type,
            'filter_fmin': filter_fmin,
            'filter_fmax': filter_fmax,
        }


    def set_time(self, dt_iter, tstart, tend):
        """
        Set parameters defining time windows over which DeepPhasePick are applied.

        Parameters
        ----------
        dt_iter: float
            time step (in seconds) between consecutive time windows.
        tstarts: str
            start time to define time windows, in format "YYYY-MM-DDTHH:MM:SS".
        tends: str
            end time to define time windows, in format "YYYY-MM-DDTHH:MM:SS".

        """
        self.time = {
            'dt_iter': dt_iter,
            'tstart': oc.UTCDateTime(tstart),
            'tend': oc.UTCDateTime(tend),
        }


    def set_trigger(self, n_shift=10, pthres_p=[0.9,.001], pthres_s=[0.9,.001], max_trig_len=[9e99, 9e99]):
        """
        Set parameters defining how predicted discrete probability time series are computed when running phase detection on seismic waveforms

        Parameters
        ----------
        n_shift: int, optional
            step size (in samples) defining discrete probability time series.
        pthres_p: list of float, optional
            probability thresholds defining P-phase trigger on (pthres_p[0]) and off (pthres_p[1]) times.
            See thres1 and thres2 parameters in obspy trigger_onset function.
        pthres_s: list of float, optional
            probability thresholds defining S-phase trigger on (pthres_s[0]) and off (pthres_s[1]) times.
            See thres1 and thres2 parameters in function obspy.signal.trigger.trigger_onset.
        max_trig_len: list of int, optional
            maximum lengths (in samples) of triggered P (max_trig_len[0]) and S (max_trig_len[1]) phase.
            See max_len parameter in function obspy.signal.trigger.trigger_onset.
        """

        self.trigger = {
            'n_shift': n_shift,
            'pthres_p': pthres_p,
            'pthres_s': pthres_s,
            'max_trig_len': max_trig_len,
        }


    def set_picking(self, op_conds=['1','2','3','4'], tp_th_add=1.5, dt_sp_near=3., dt_ps_max=25., dt_sdup_max=3., run_mcd=True, mcd_iter=10):
        """
        Set parameters applied in optional conditions for refining preliminary picks obtained from phase detection.

        Parameters
        ----------
        op_conds: list of str, optional
            optional conditions that are applied on preliminary picks, in order to keep keep/remove presumed true/false preliminary onsets.
            These conditions are explained in Supplementary Information of the original publication (https://doi.org/10.31223/X5BC8B).
            For example ['1', '2'] indicates that only conditions (1) and (2) are applied.
            '1': resolves between P and S phases predicted close in time, with overlapping probability time series
            '2': resolves between P and S phases predicted close in time, with no overlapping probability distributions.
            '3': discards S picks for which there is no earlier P or P-S predicted picks.
            '4': resolves between possible duplicated S phases.
        tp_th_add: float, optional
            time (in seconds) added to define search time intervals in condition (1).
        dt_sp_near: float, optional
            time threshold (in seconds) used in condition (2).
        dt_ps_max: float, optional
            time (in seconds) used to define search time intervals in condition (3).
        dt_sdup_max: float, optional
            time threshold (in seconds) used in condition (4).
        run_mcd: bool, optional
            If True, run phase picking in order to refine preliminary picks from phase detection.
        mcd_iter: int, optional
            number of Monte Carlo Dropout iterations used in phase picking.
        """

        self.picking = {
            'op_conds': op_conds,
            'tp_th_add': tp_th_add,
            'dt_sp_near': dt_sp_near,
            'dt_ps_max': dt_ps_max,
            'dt_sdup_max': dt_sdup_max,
            'run_mcd': run_mcd,
            'mcd_iter': mcd_iter,
        }
