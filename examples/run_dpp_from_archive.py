#!/usr/bin/env python
#
# This script applies DeepPhasePick on seismic data stored in a archive directory structured as: archive/YY/NET/STA/CH
# Here YY is year, NET is the network code, STA is the station code and CH is the channel code (e.g., HH) of the seismic streams.
#
# Author: Hugo Soto Parada (June, 2021)
# Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com
#
########################################################################################################################################

import os
import config, data, model, util

# 1. Configure DPP
#
# config
util.init_session()
dpp_config = config.Config()
dpp_config.set_trigger(pthres_p=[0.9, 0.001], pthres_s=[0.9, 0.001])
# dpp_config.set_picking(mcd_iter=10, run_mcd=True)
dpp_config.set_picking(run_mcd=False)
#
dpp_config.set_data(
    stas=['PB01', 'PB02'],
    net='CX',
    ch='HH',
    archive='sample_data/archive',
    opath='out_archive'
)
dpp_config.set_time(
    dt_iter=3600.,
    tstart="2014-05-01T00:00:00",
    tend="2014-05-01T12:00:00",
)

# 2. Read seismic data into DPP
#
# data
dpp_data = data.Data()
dpp_data.read_from_archive(dpp_config)
#
for k in dpp_data.data:
    print(k, dpp_data.data[k])

# 3. Run phase detection and picking
#
# model
dpp_model = model.Model(verbose=False)
# dpp_model = model.Model(verbose=False, version_pick_P="20201002_2", version_pick_S="20201002_2")
#
print(dpp_model.model_detection['best_model'].summary())
print(dpp_model.model_picking_P['best_model'].summary())
print(dpp_model.model_picking_S['best_model'].summary())
#
# run phase detection
dpp_model.run_detection(dpp_config, dpp_data, save_dets=False, save_data=False)
#
# run phase picking
dpp_model.run_picking(dpp_config, dpp_data, save_plots=False, save_stats=True, save_picks=False)

# 4. Plot predicted phases
#
# plots
# util.plot_predicted_phases(dpp_config, dpp_data, dpp_model)
util.plot_predicted_phases(dpp_config, dpp_data, dpp_model, plot_probs=['P','S'], shift_probs=True)

