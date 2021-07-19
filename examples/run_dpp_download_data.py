#!/usr/bin/env python
#
# This script applies DeepPhasePick on seismic data downloaded using FDSN web service client for ObsPy.
#
# Author: Hugo Soto Parada (June, 2021)
# Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com
#
########################################################################################################################################

import os
import config, data, model, util
from obspy.clients.fdsn import Client
import obspy.core as oc

# 1. Configure DPP
#
# config
util.init_session()
dpp_config = config.Config()
dpp_config.set_trigger(pthres_p=[0.9, 0.001], pthres_s=[0.9, 0.001])
dpp_config.set_picking(mcd_iter=10, run_mcd=True)
# dpp_config.set_picking(run_mcd=False)
#
dpp_config.set_data(
    stas=['PB01', 'PB02'],
    net='CX',
    ch='HH',
    archive='sample_data/CX_20140401',
    opath='out_CX_20140401'
)
dpp_config.set_time(
    dt_iter=3600.,
    tstart="2014-04-01T02:00:00",
    tend="2014-04-01T03:00:00",
)

# 2. Download seismic data and read it into DPP
#
# download and archive seismic waveforms
client = Client("GFZ")
os.makedirs(f"{dpp_config.data['archive']}", exist_ok=True)
tstart = oc.UTCDateTime(dpp_config.time['tstart'])
tend = oc.UTCDateTime(dpp_config.time['tend'])
#
for sta in dpp_config.data['stas']:
    st = client.get_waveforms(network=dpp_config.data['net'], station=sta, location="*", channel=f"{dpp_config.data['ch']}?", starttime=tstart, endtime=tend)
    # print(st)
    st_name = f"{dpp_config.data['archive']}/{st[0].stats.network}.{st[0].stats.station}..{st[0].stats.channel[:-1]}.mseed"
    print(f"writing stream {st_name}...")
    st.write(st_name, format="MSEED")
#
# data
dpp_data = data.Data()
dpp_data.read_from_directory(dpp_config)
#
# for k in dpp_data.data:
#     print(k, dpp_data.data[k])

# 3. Run phase detection and picking
#
# model
# dpp_model = model.Model(verbose=False)
# dpp_model = model.Model(verbose=False, version_pick_P="20201002_2", ntrials_P=50)
dpp_model = model.Model(verbose=False, version_pick_P="20201002_2", ntrials_P=50, version_pick_S="20201002_2", ntrials_S=50)
#
# print(dpp_model.model_detection['best_model'].summary())
# print(dpp_model.model_picking_P['best_model'].summary())
# print(dpp_model.model_picking_S['best_model'].summary())
#
# run phase detection
dpp_model.run_detection(dpp_config, dpp_data, save_dets=False, save_data=False)
#
# run phase picking
dpp_model.run_picking(dpp_config, dpp_data, save_plots=True, save_stats=True, save_picks=False)

# 4. Plot predicted phases
#
# plots
util.plot_predicted_phases(dpp_config, dpp_data, dpp_model)
# util.plot_predicted_phases(dpp_config, dpp_data, dpp_model, plot_probs=['P','S'], shift_probs=True)

