# coding: utf-8

"""
This module contains a class and methods related to the seismic data used by DeepPhasePick method.

Author: Hugo Soto Parada (October, 2020)
Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com

"""

import numpy as np
import obspy.core as oc
from obspy.signal.trigger import trigger_onset
from obspy.io.mseed.core import InternalMSEEDError
from glob import glob
import re, sys, os, shutil, gc


class Data():
    """
    Class defining seismic data-related methods.
    """

    def read_from_archive(self, config):
        """
        Reads seismic data on which DeepPhasePick is applied.
        Data must be stored in a archive directory structured as: archive/YY/NET/STA/CH
        Here YY is year, NET is the network code, STA is the station code and CH is the channel code (e.g., HH) of the seismic streams.

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration defining which seismic waveform data is selected and how this data is processed in DeepPhasePick.
        """
        #
        # set time windows for iteration over continuous waveforms
        #
        tstart, tend, dt_iter = [config.time['tstart'], config.time['tend'], config.time['dt_iter']]
        t_iters = []
        tstart_iter = tstart
        tend_iter = tstart_iter + dt_iter
        if tend_iter > tend:
            tend_iter = tend
        #
        print("#")
        while tstart_iter < tend:
            t_iters.append([tstart_iter, tend_iter])
            tstart_iter += dt_iter
            tend_iter += dt_iter
            if tend_iter > tend:
                tend_iter = tend
            # print(t_iters[-1], tstart_iter, tend_iter)
        #
        print(f"time windows ({len(t_iters)}) for iteration over continuous waveforms:")
        for t_iter in t_iters:
            print(t_iter)
        print("")
        #
        # iterate over time windows
        #
        self.data = {}
        stas = config.data['stas']
        doy_tmp = '999'
        for i, t_iter in enumerate(t_iters):
            #
            tstart_iter, tend_iter = t_iter
            #
            twin_str = f"{tstart_iter.year}{tstart_iter.month:02}{tstart_iter.day:02}"
            twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}{tstart_iter.second:02}"
            twin_str += f"_{tend_iter.year}{tend_iter.month:02}{tend_iter.day:02}"
            twin_str += f"T{(tend_iter).hour:02}{(tend_iter).minute:02}{(tend_iter).second:02}"
            opath = f"{config.data['opath']}/{twin_str}"
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
                    # read seismic waveforms from archive
                    #
                    net = config.data['net']
                    ch = config.data['ch']
                    path = config.data['archive']
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
                if config.data_params['st_detrend']:
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
                if config.data_params['st_filter'] is not None:
                    #
                    print('filter...')
                    st.filter(type=config.data_params['st_filter'], **config.data_params['filter_opts'])
                #
                if config.data_params['st_resample']:
                    #
                    print('resampling...')
                    for tr in st:
                        #
                        if tr.stats.sampling_rate == config.data_params['samp_freq']:
                            outstr = f"{tr} --> skipped, already sampled at {tr.stats.sampling_rate} Hz"
                            print(outstr)
                            pass
                        #
                        if tr.stats.sampling_rate < config.data_params['samp_freq']:
                            outstr = f"{tr} --> resampling from {tr.stats.sampling_rate} to {config.data_params['samp_freq']} Hz"
                            print(outstr)
                            tr.resample(config.data_params['samp_freq'])
                        #
                        if tr.stats.sampling_rate > config.data_params['samp_freq']:
                            #
                            if int(tr.stats.sampling_rate % config.data_params['samp_freq']) == 0:
                                decim_factor = int(tr.stats.sampling_rate / config.data_params['samp_freq'])
                                outstr = f"{tr} --> decimating from {tr.stats.sampling_rate} to {config.data_params['samp_freq']} Hz"
                                print(outstr)
                                tr.decimate(decim_factor)
                            else:
                                outstr = f"{tr} --> resampling from {tr.stats.sampling_rate} to {config.data_params['samp_freq']} Hz !!"
                                print(outstr)
                                tr.resample(config.data_params['samp_freq'])
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
            # sort traces of same station by channel, so for each station traces will be shown in order (HHE,N,Z)
            stt.sort(['channel'])
            # print(stt.__str__(extended=True))
            #
            self.data[i+1] = {
                'st': {},
                'twin': [tstart_iter, tend_iter],
                'opath': opath,
            }
            for t, tr in enumerate(stt):
                sta = tr.stats.station
                #
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


    def read_from_directory(self, config):
        """
        Reads seismic data on which DeepPhasePick is applied.
        All waveforms must be stored in an unstructured archive directory, e.g.: archive/

        Parameters
        ----------
        config: instance of config.Config
            Contains user configuration defining which seismic waveform data is selected and how this data is processed in DeepPhasePick.
        """
        #
        # read seismic waveforms from directory
        #
        path = config.data['archive']
        tstart, tend, dt_iter = [config.time['tstart'], config.time['tend'], config.time['dt_iter']]
        flist_all = sorted(glob(path+'/*'))[:]
        #
        flabels = []
        sts = [oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream(), oc.Stream()]
        st = oc.Stream()
        st_arg = 0
        for i, f in enumerate(flist_all[:]):
            #
            if len(sts[st_arg]) >= 50:
                st_arg += 1
            #
            print(f"reading seismic waveform: {f}")
            tr = oc.read(f)
            tr_net, tr_sta, tr_ch = tr[0].stats.network, tr[0].stats.station, tr[0].stats.channel[:-1]
            if (tr_net == config.data['net']) and (tr_sta in config.data['stas']) and (tr_ch == config.data['ch']):
                try:
                    sts[st_arg] += tr
                except InternalMSEEDError:
                    # stas_remove.append(sta)
                    outstr = f"skipping {f} --> InternalMSEEDError exception"
                    print(outstr)
                    continue
        #
        for stt in sts:
            st += stt
        del sts
        #
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
        if config.data_params['st_detrend']:
            #
            print('detrend...')
            try:
                st.detrend(type='linear')
            except NotImplementedError:
                #
                # Catch exception NotImplementedError: Trace with masked values found. This is not supported for this operation.
                # Try split() method on Stream to produce a Stream with unmasked Traces.
                #
                st = st.split()
                st.detrend(type='linear')
            except ValueError:
                #
                # Catch exception ValueError: array must not contain infs or NaNs.
                # Due to presence e.g. of NaNs in at least one trace data.
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
        if config.data_params['st_filter'] is not None:
            #
            print('filter...')
            st.filter(type=config.data_params['st_filter'], **config.data_params['filter_opts'])
        #
        if config.data_params['st_resample']:
            #
            print('resampling...')
            for tr in st:
                #
                if tr.stats.sampling_rate == config.data_params['samp_freq']:
                    outstr = f"{tr} --> skipped, already sampled at {tr.stats.sampling_rate} Hz"
                    print(outstr)
                    pass
                #
                if tr.stats.sampling_rate < config.data_params['samp_freq']:
                    outstr = f"{tr} --> resampling from {tr.stats.sampling_rate} to {config.data_params['samp_freq']} Hz"
                    print(outstr)
                    tr.resample(config.data_params['samp_freq'])
                #
                if tr.stats.sampling_rate > config.data_params['samp_freq']:
                    #
                    if int(tr.stats.sampling_rate % config.data_params['samp_freq']) == 0:
                        decim_factor = int(tr.stats.sampling_rate / config.data_params['samp_freq'])
                        outstr = f"{tr} --> decimating from {tr.stats.sampling_rate} to {config.data_params['samp_freq']} Hz"
                        print(outstr)
                        tr.decimate(decim_factor)
                    else:
                        outstr = f"{tr} --> resampling from {tr.stats.sampling_rate} to {config.data_params['samp_freq']} Hz !!"
                        print(outstr)
                        tr.resample(config.data_params['samp_freq'])
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
        # set time windows for iteration over continuous waveforms
        #
        t_iters = []
        tstart_iter = tstart
        tend_iter = tstart_iter + dt_iter
        if tend_iter > tend:
            tend_iter = tend
        #
        print("#")
        while tstart_iter < tend:
            t_iters.append([tstart_iter, tend_iter])
            tstart_iter += dt_iter
            tend_iter += dt_iter
            if tend_iter > tend:
                tend_iter = tend
            # print(t_iters[-1], tstart_iter, tend_iter)
        #
        print(f"time windows ({len(t_iters)}) for iteration over continuous waveforms:")
        for t_iter in t_iters:
            print(t_iter)
        print("")
        #
        # iterate over time windows
        #
        self.data = {}
        for i, t_iter in enumerate(t_iters):
            #
            tstart_iter, tend_iter = t_iter
            #
            twin_str = f"{tstart_iter.year}{tstart_iter.month:02}{tstart_iter.day:02}"
            twin_str += f"T{tstart_iter.hour:02}{tstart_iter.minute:02}{tstart_iter.second:02}"
            twin_str += f"_{tend_iter.year}{tend_iter.month:02}{tend_iter.day:02}"
            twin_str += f"T{(tend_iter).hour:02}{(tend_iter).minute:02}{(tend_iter).second:02}"
            #
            opath = f"{config.data['opath']}/{twin_str}"
            yy = tstart_iter.year
            doy = '%03d' % (tstart_iter.julday)
            #
            print('slicing...')
            stt = st.slice(tstart_iter, tend_iter)
            #
            # sort traces of same station by channel, so for each station traces will be shown in order (HHE,N,Z)
            stt.sort(['channel'])
            # print(stt.__str__(extended=True))
            #
            self.data[i+1] = {
                'st': {},
                'twin': [tstart_iter, tend_iter],
                'opath': opath,
            }
            for t, tr in enumerate(stt):
                sta = tr.stats.station
                #
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
