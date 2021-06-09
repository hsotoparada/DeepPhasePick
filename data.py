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

import numpy as np
import obspy.core as oc
from obspy.signal.trigger import trigger_onset
from obspy.io.mseed.core import InternalMSEEDError
from glob import glob
import re, sys, os, shutil, gc


class Data():
    """
    Retrieve seismic data...
    """


    # def get_data_time(self, best_model, best_params, dct_time, dct_sta, dct_param, dct_out, t):
    def get_from_archive(self, config):
        """
        Performs P- and S-phase detection task.
        Returns a dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
        ----------
        best_model: best (keras) model trained for phase detection.
        best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
        dct_time: (list) dictionary defining time windows over which prediction is performed.
        dct_sta: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_out: (dict) dictionary defining DeepPhasePick output options.
        t: (int) index number of the waveform time window on which prediction is performed.
        """
        #
        # flag_data = dct_out['flag_data']
        # tstart, tend, dt_iter = pred_times
        # tstart, tend, dt_iter = [dct_time[flag_data]['tstarts'][t], dct_time[flag_data]['tends'][t], dct_time[flag_data]['dt_iter']]
        tstart, tend, dt_iter = [config.time['tstart'], config.time['tend'], config.time['dt_iter']]
        #
        t_iters = []
        tstart_iter = tstart
        tend_iter = tstart_iter + dt_iter
        #
        # print("parameters for waveform processing:")
        # print([f"{k}: {dct_param[k]}" for k in dct_param])
        # #
        print("#")
        while tstart_iter < tend:
            t_iters.append([tstart_iter, tend_iter])
            tstart_iter += dt_iter
            tend_iter += dt_iter
        #
        print(f"preparing time windows ({len(t_iters)}) for iteration over continuous waveforms...")
        for t_iter in t_iters:
            print(t_iter)
        print("")
        #
        # dct_data = {}
        self.data = {}
        stas = config.data['stas']
        # stas = dct_sta[flag_data]['stas']
        doy_tmp = '999'
        for i, t_iter in enumerate(t_iters):
            #
            tstart_iter, tend_iter = t_iter
            #
            twin_str = f"{tstart_iter.year}{tstart_iter.month:02}{tstart_iter.day:02}"
            # twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}"
            # twin_str += f"_{(tend_iter-1.).hour:02}{(tend_iter-1.).minute:02}"
            twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}{tstart_iter.second:02}"
            twin_str += f"_{(tend_iter-1.).hour:02}{(tend_iter-1.).minute:02}{(tend_iter-1.).second:02}"
            # opath = f"{dct_out['opath']}/{flag_data}/wf_{twin_str}"
            # opath = f"{dct_out['opath']}/{flag_data}/wfs/wf_{twin_str}"
            opath = f"{config.data['opath']}/wfs/wf_{twin_str}"
            yy = tstart_iter.year
            doy = '%03d' % (tstart_iter.julday)
            #
            if doy != doy_tmp:
                # print(t_iter)
                # print(tstart_iter, tend_iter)
                sts = [oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream()]
                st = oc.Stream()
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
                    net = config.data['net']
                    ch = config.data['ch']
                    path = config.data['archive']
                    # net = dct_sta[flag_data]['net']
                    # ch = dct_sta[flag_data]['ch']
                    # path = dct_sta[flag_data]['archive']
                    # flist = glob('archive/'+str(yy)+'/'+net+'/'+sta+'/'+ch+'?.D/*.'+doy)
                    flist = glob(path+'/'+str(yy)+'/'+net+'/'+sta+'/'+ch+'?.D/*.'+doy)
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
                        try:
                            print(f)
                            tr = oc.read(f)
                            # print(tr)
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
                # process (detrend, filter, resample, ...) raw stream data
                #
                print("#")
                print("processing raw stream data...")
                #
                # stt = st.copy()
                # del st
                #
                if config.params['st_detrend']:
                # if dct_param['st_detrend']:
                    #
                    print('detrend...')
                    try:
                        st.detrend(type='linear')
                    except NotImplementedError:
                        #
                        # Catch exception NotImplementedError: Trace with masked values found. This is not supported for this operation.
                        # Try the split() method on Stream to produce a Stream with unmasked Traces.
                        #
                        st = st.split()
                        st.detrend(type='linear')
                    except ValueError:
                        #
                        # Catch exception ValueError: array must not contain infs or NaNs.
                        # Due to presence of e.g. nans in at least one trace data.
                        #
                        stas_remove = []
                        for tr in st:
                            nnan = np.count_nonzero(np.isnan(tr.data))
                            ninf = np.count_nonzero(np.isinf(tr.data))
                            if nnan > 0:
                                print(f"{tr} --> removed (due to presence of nans)")
                                stas_remove.append(tr.stats.station)
                                continue
                            if ninf > 0:
                                print(f"{tr} --> removed (due to presence of infs)")
                                stas_remove.append(tr.stats.station)
                                continue
                        #
                        stas_remove = set(stas_remove)
                        for s in stas_remove:
                            for tr in st.select(station=s):
                                st.remove(tr)
                        st.detrend(type='linear')
                #
                if config.params['st_filter']:
                    #
                    print('filter...')
                    if config.params['filter'] == 'bandpass':
                        st.filter(type=config.params['filter'], freqmin=config.params['filter_freq_min'], freqmax=config.params['filter_freq_max'])
                    elif config.params['filter'] == 'highpass':
                        st.filter(type=config.params['filter'], freq=config.params['filter_freq_min'])
                #
                print('resampling...')
                for tr in st:
                    if tr.stats.sampling_rate < config.params['samp_freq']:
                        outstr = f"{tr} --> resampling from {tr.stats.sampling_rate} to {config.params['samp_freq']} Hz"
                        print(outstr)
                        tr.resample(config.params['samp_freq'])
                    #
                    # conditions for cases with sampling rate higher than samp_freq and different than 200 Hz should be included here
                    if tr.stats.sampling_rate == 200.:
                        outstr = f"{tr} --> decimating from {tr.stats.sampling_rate} to {config.params['samp_freq']} Hz"
                        print(outstr)
                        tr.decimate(2)
                #
                print('merging...')
                try:
                    st.merge()
                except:
                    #
                    outstr = f"Catch exception: can't merge traces with same ids but differing data types!"
                    print(outstr)
                    for tr in st:
                        tr.data = tr.data.astype(np.int32)
                    st.merge()
            #
            print('slicing...')
            stt = st.slice(tstart_iter, tend_iter)
            #
            self.data[i+1] = {
                'st': {},
                'twin': [tstart_iter, tend_iter],
                'opath': opath,
            }
            for t, tr in enumerate(stt):
                sta = tr.stats.station
                #
                # TODO: update this in final dpp.py
                st_check = stt.select(station=sta)
                if len(st_check) < 3:
                    print(f"skipping trace of stream with less than 3 components: {tr}")
                    continue
                #
                if sta not in self.data[i+1]['st'].keys():
                    self.data[i+1]['st'][sta] = oc.Stream()
                    self.data[i+1]['st'][sta] += tr
                else:
                    self.data[i+1]['st'][sta] += tr
            #
            doy_tmp = doy
        #
        # return self.data


    # TODO: filter out samples out of selected time range
    # def get_data_2(best_model, best_params, dct_sta, dct_param, dct_out):
    def get_from_samples(self, config):
        """
        Retrieve seismic data...
        ----------
        dct_sta: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
        dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
        dct_out: (dict) dictionary defining DeepPhasePick output options.
        t: (int) index number of the waveform time window on which prediction is performed.
        """
        #
        # t_iters = []
        # flag_data = dct_out['flag_data']
        #
        # # path = "/home/soto/Work/ML/git/EQTransformer/test_dpp_picking/samples_max_100/mseed_ev"
        # path = "/home/soto/Work/ML/git/EQTransformer/test_dpp_detection/samples_max_100/mseed_ev"
        # flist_all = sorted(glob(path+'/*/*'))
        #
        path = config.data['archive']
        # path = "/home/soto/Work/ML/git/EQTransformer/test_dpp_detection/samples_all/mseed_ev_paper_rev"
        # path = "/home/soto/Work/ML/git/EQTransformer/test_dpp_picking/samples_all/mseed_ev"
        flist_tmp = sorted(glob(path+'/*'))[:]
        # flist_tmp = sorted(glob(path+'/*'))[:25]
        flist_all = []
        for f_tmp in flist_tmp:
            flist_all.extend(sorted(glob(f_tmp+'/*')))
        #
        #
        flabels = []
        # dct_data = {}
        self.data = {}
        ii = 0
        for i, f in enumerate(flist_all[:]):
            #
            tstart_str = f.split('/')[-1].split('__')[1].split('Z')[0]
            tstart_iter = oc.UTCDateTime(tstart_str)
            tend_str = f.split('/')[-1].split('__')[2].split('Z')[0]
            tend_iter = oc.UTCDateTime(tend_str)
            net = f.split('/')[-1].split('.')[0]
            sta = f.split('/')[-1].split('.')[1]
            #
            twin_str = f"{tstart_iter.year}{tstart_iter.month:02}{tstart_iter.day:02}"
            # twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}"
            # twin_str += f"_{(tend_iter-1.).hour:02}{(tend_iter-1.).minute:02}"
            twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}{tstart_iter.second:02}"
            twin_str += f"_{(tend_iter).hour:02}{(tend_iter).minute:02}{(tend_iter).second:02}"
            # opath = f"{dct_out['opath']}/{flag_data}/wf_{twin_str}"
            # opath = f"{dct_out['opath']}/{flag_data}/wf_{twin_str}_{net}_{sta}"
            opath = f"{config.data['opath']}/wfs/wf_{twin_str}_{net}_{sta}"
            yy = tstart_iter.year
            # doy = '%03d' % (tstart_iter.julday)
            #
            ###
            # print(t_iter)
            # print(tstart_iter, tend_iter)
            # sts = [oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream()]
            st = oc.Stream()
            #
            # st_arg = 0
            stas_remove = []
            #
            ch = f.split('/')[-1].split('.')[3].split('__')[0]
            # path = dct_sta[flag_data]['archive']
            # flist = glob('archive/'+str(yy)+'/'+net+'/'+sta+'/'+ch+'?.D/*.'+doy)
            flist = glob(f"{path}/{sta}/{net}.{sta}*__{tstart_str}*{tend_str}*")
            flabel = f"{net}.{sta}*__{tstart_str}*{tend_str}*"
            #
            if flabel not in flabels:
                #
                print("")
                print("retrieving seismic waveforms for stations:")
                # print(stas)
                #
                flabels.append(flabel)
                ii += 1
                if len(flist) > 0:
                    # outstr = f"seismic data found for: net = {net}, sta = {sta}, st_count = {len(sts[st_arg])}, st_arg = {st_arg}"
                    outstr = f"seismic data found for: net = {net}, sta = {sta}"
                    print(outstr)
                    outstr = str(flist)
                    print(outstr)
                else:
                    outstr = f"seismic data not found for: net = {net}, sta = {sta}"
                    print(outstr)
                #
                # if len(sts[st_arg]) >= 50:
                #     st_arg += 1
                #
                for f in flist:
                    try:
                        print(f)
                        tr = oc.read(f)
                        # print(tr)
                        # sts[st_arg] += tr
                        st += tr
                    #
                    except InternalMSEEDError:
                        stas_remove.append(sta)
                        outstr = f"skipping {f} --> InternalMSEEDError exception"
                        print(outstr)
                        continue
                #
                # for stt in sts:
                #     st += stt
                # del sts
                #
                stas_remove = set(stas_remove)
                # print(stas_remove)
                for s in stas_remove:
                    for tr in st.select(station=s):
                        st.remove(tr)
                # print(len(st))
                print(st.__str__(extended=True))
                #
                # process (detrend, filter, resample, ...) raw stream data
                #
                print("#")
                print("processing raw stream data...")
                #
                # stt = st.copy()
                # del st
                #
                # if dct_param['st_detrend']:
                if config.params['st_detrend']:
                    #
                    print('detrend...')
                    try:
                        st.detrend(type='linear')
                    except NotImplementedError:
                        #
                        # Catch exception NotImplementedError: Trace with masked values found. This is not supported for this operation.
                        # Try the split() method on Stream to produce a Stream with unmasked Traces.
                        #
                        st = st.split()
                        st.detrend(type='linear')
                    except ValueError:
                        #
                        # Catch exception ValueError: array must not contain infs or NaNs.
                        # Due to presence of e.g. nans in at least one trace data.
                        #
                        stas_remove = []
                        for tr in st:
                            nnan = np.count_nonzero(np.isnan(tr.data))
                            ninf = np.count_nonzero(np.isinf(tr.data))
                            if nnan > 0:
                                print(f"{tr} --> removed (due to presence of nans)")
                                stas_remove.append(tr.stats.station)
                                continue
                            if ninf > 0:
                                print(f"{tr} --> removed (due to presence of infs)")
                                stas_remove.append(tr.stats.station)
                                continue
                        #
                        stas_remove = set(stas_remove)
                        for s in stas_remove:
                            for tr in st.select(station=s):
                                st.remove(tr)
                        st.detrend(type='linear')
                #
                if config.params['st_filter']:
                    #
                    print('filter...')
                    if config.params['filter'] == 'bandpass':
                        st.filter(type=config.params['filter'], freqmin=config.params['filter_freq_min'], freqmax=config.params['filter_freq_max'])
                    elif config.params['filter'] == 'highpass':
                        st.filter(type=config.params['filter'], freq=config.params['filter_freq_min'])
                #
                print('resampling...')
                for tr in st:
                    if tr.stats.sampling_rate < config.params['samp_freq']:
                        outstr = f"{tr} --> resampling from {tr.stats.sampling_rate} to {config.params['samp_freq']} Hz"
                        print(outstr)
                        tr.resample(config.params['samp_freq'])
                    #
                    # conditions for cases with sampling rate higher than samp_freq and different than 200 Hz should be included here
                    if tr.stats.sampling_rate == 200.:
                        outstr = f"{tr} --> decimating from {tr.stats.sampling_rate} to {config.params['samp_freq']} Hz"
                        print(outstr)
                        tr.decimate(2)
                #
                print('merging...')
                try:
                    st.merge()
                except:
                    #
                    outstr = f"Catch exception: can't merge traces with same ids but differing data types!"
                    print(outstr)
                    for tr in st:
                        tr.data = tr.data.astype(np.int32)
                    st.merge()
                # #
                # print('slicing...')
                # stt = st.slice(tstart_iter, tend_iter)
                #
                self.data[ii] = {
                    'st': {},
                    'twin': [tstart_iter, tend_iter],
                    'opath': opath,
                    'ipath': f,
                }
                # self.data[i+1] = {
                # 	'st': {}, 'pred': {},
                # 	'twin': [tstart_iter, tend_iter],
                # }
                for t, tr in enumerate(st):
                    sta = tr.stats.station
                    if sta not in self.data[ii]['st'].keys():
                        self.data[ii]['st'][sta] = oc.Stream()
                        self.data[ii]['st'][sta] += tr
                    else:
                        self.data[ii]['st'][sta] += tr
                #
                # doy_tmp = doy
        #
        # return self.data


