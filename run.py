#!/usr/bin/env python
#
# This scripts runs DeepPhasePick, a method for automatically detecting and picking seismic phases
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

import config, data, model, util
from datetime import datetime
# import os, sys, gc


########################################################################################################################
#
tep_s = datetime.now()
#
## RUN ##
#
# config
user_config = config.Config()
# user_config.set_params(st_filter=True)
user_config.set_trigger(pthres_p=[0.9, 0.001], pthres_s=[0.95, 0.001])
user_config.set_picking(op_conds=[0,1,2,3], dt_PS_max=40., dt_sdup_max=5., mcd_iter=10)
user_config.set_data(
    stas=['AB01'],
    net='9K',
    ch='HH',
    archive='/home/soto/Volumes/CHILE/soto_work/HART_ALBANIA/archive',
    opath='out'
)
user_config.set_time(
    dt_iter=3600.,
    tstart= "2020-01-11T21:00:00",
    tend= "2020-01-11T23:00:00",
    # tend= "2020-01-12T02:00:00",
)
#
# data
user_data = data.Data()
user_data.get_from_archive(user_config)
# user_data.get_from_samples(user_config)
#
# model
user_model = model.Model()
user_model.run_detection(user_config, user_data, user_model)
user_model.run_picking(user_config, user_data, user_model, save_plot=True, save_stat=True)
# print(user_model.model_detection['best_model'].summary())
# print(user_model.model_picking_P['best_model'].summary())
# print(user_model.model_picking_S['best_model'].summary())
#
# plots
util.plot_predicted_wf_phases(user_config, user_data, user_model)
# util.plot_predicted_wf_phases_prob(fmt, comps)
util.plotly_predicted_wf_phases(user_config, user_data, user_model)
#
# elapsed time
#
tep_e = datetime.now()
time_exec = tep_e - tep_s
log_out = f"Process completed -- Elapsed time: {time_exec}"
print('')
print(log_out)


