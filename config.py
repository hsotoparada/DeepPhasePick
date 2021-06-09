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

import obspy.core as oc
from datetime import datetime
import re, sys, os, shutil, gc


class Config():
    """
    Initiate user configuration...
    """

    def __init__(self, dct_params=None, dct_trigger=None, dct_picking=None, dct_data=None, dct_time=None):
        """
        ...
        """

        self.params = self._set_default_params(dct_params)
        self.trigger = self._set_default_trigger(dct_trigger)
        self.picking = self._set_default_picking(dct_picking)
        self.data = self._set_default_data(dct_data)
        self.time = self._set_default_time(dct_time)


    def _set_default_params(self, dct_params):
        """
        Set ...
        """

        dct = {
            'samp_freq': 100.,
            # 'samp_dt': 1 / samp_freq,
            'st_normalized': True,
            'st_detrend': True,
            'st_filter': False,
            'filter': 'bandpass',
            # 'filter': 'bandpass',
            'filter_freq_min': .2,
            'filter_freq_max': 10.,
        }

        if dct_params is not None:
            for k in dct:
                if k in dct_params:
                    dct[k] = dct_params[k]

        return dct


    def _set_default_trigger(self, dct_trigger):
        """
        Set ...
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
        Set ...
        """

        dct = {
            # 'op_conds': ['1', '2', '3', '4'],
            'op_conds': [],
            #
            'dt_PS_max': 35.,
            'dt_sdup_max': 3.,
            'dt_sp_near': 3.,
            'tp_th_add': 1.5,
            #
            # 'dt_PS_max': 25.,
            # 'dt_sdup_max': 2.,
            # 'dt_sp_near': 1.5,
            # 'tp_th_add': 1.5,
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


    def _set_default_data(self, dct_data):
        """
        Set ...
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


    def _set_default_time(self, dct_time):
        """
        Set ...
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


    def set_params(self, samp_freq=100., st_normalized=True, st_detrend=True, st_filter=False, st_filter_type='bandpass', filter_fmin=.2, filter_fmax=10.):
        """
        Set ...
        """

        self.params = {
            'samp_freq': samp_freq,
            'st_normalized': st_normalized,
            'st_detrend': st_detrend,
            'st_filter': st_filter,
            'st_filter_type': st_filter_type,
            'filter_fmin': filter_fmin,
            'filter_fmax': filter_fmax,
        }


    def set_trigger(self, n_shift=10., pthres_p=[0.9,.001], pthres_s=[0.9,.001], max_trig_len=[9e99, 9e99]):
        """
        Set ...
        """

        self.trigger = {
            'n_shift': n_shift,
            'pthres_p': pthres_p,
            'pthres_s': pthres_s,
            'max_trig_len': max_trig_len,
        }


    def set_picking(self, op_conds=[], dt_PS_max=35., dt_sdup_max=3., dt_sp_near=3., tp_th_add=1.5, run_mcd=True, mcd_iter=10):
        """
        Set ...
        """

        self.picking = {
            'op_conds': op_conds,
            'dt_PS_max': dt_PS_max,
            'dt_sdup_max': dt_sdup_max,
            'dt_sp_near': dt_sp_near,
            'tp_th_add': tp_th_add,
            'run_mcd': run_mcd,
            'mcd_iter': mcd_iter,
        }


    # TODO: set default args to sample data
    def set_data(self, stas, ch, net, archive, opath):
        """
        Set ...
        """
        self.data = {
            'stas': stas,
            'ch': ch,
            'net': net,
            'archive': archive,
            'opath': opath,
        }


    # TODO: set default args to sample data
    def set_time(self, dt_iter, tstart, tend):
        """
        Set ...
        """
        self.time = {
            'dt_iter': dt_iter,
            'tstart': oc.UTCDateTime(tstart),
            'tend': oc.UTCDateTime(tend),
        }


