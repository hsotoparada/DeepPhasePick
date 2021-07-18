#!/usr/bin/env python
#
# This script applies DeepPhasePick on seismic data stored in an unstructured archive directory.
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
# dpp_config.set_picking(op_conds=['1','2','3','4'], tp_th_add=1.5, dt_sp_near=1.5, dt_ps_max=25., dt_sdup_max=2., mcd_iter=10, run_mcd=True)
dpp_config.set_picking(op_conds=['1','2','3','4'], tp_th_add=1.5, dt_sp_near=1.5, dt_ps_max=25., dt_sdup_max=2., mcd_iter=10, run_mcd=False)
#
dpp_config.set_data(
    stas=['PB01', 'PB02'],
    net='CX',
    ch='HH',
    archive='sample_data/CX_20140301',
    opath='out_CX_20140301'
)
dpp_config.set_time(
    dt_iter=3600.,
    tstart="2014-03-01T02:00:00",
    tend="2014-03-01T03:00:00",
)

# 2. Read seismic data into DPP
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
dpp_model = model.Model(verbose=False, version_pick_P="20201002_2", ntrials_P=17)
#
# TODO: remove these lines
# dpp_model = model.Model(verbose=False, version_det="20210701", ntrials_det=1000)
# dpp_model = model.Model(verbose=False, version_det="20210701", ntrials_det=1000, version_pick_P="20210701", ntrials_P=17)
#
print(dpp_model.model_detection['best_model'].summary())
print(dpp_model.model_picking_P['best_model'].summary())
print(dpp_model.model_picking_S['best_model'].summary())
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

