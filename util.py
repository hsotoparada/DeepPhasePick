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
import matplotlib as mpl
import matplotlib.ticker as ticker
import pylab as plt
import tensorflow as tf
import re, sys, os, shutil, gc
import pickle


def export_dict2pckl(dct, opath):
    """
    Exports dictionary as pickle file.
    ----------
    dct: (dict) input dictionary
    opath: (str) output path to created pickle file.
    """
    with open(opath, 'wb') as pout:
        pickle.dump(dct, pout)


def import_pckl2dict(ipath):
    """
    Imports pickle file to dictionary and returns this dictionary.
    ----------
    ipath: (str) path to pickle file.
    """
    with open(ipath, 'rb') as pin:
        dct = pickle.load(pin)
    return dct


# TODO: needed
def init_session():
    """
    Sets up tensorflow v2.x / keras session.
    """
    #
    # This is to avoid error:
    # Failed to get convolution algorithm. This is probably because cuDNN failed to initialize...
    config = tf.compat.v1.ConfigProto()
    # config = tf.ConfigProto()
    #
    # It allows any new GPU process which consumes a GPU memory to be run on the same machine.
    # see: https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # tf.keras.backend.set_session(tf.Session(config=config))
    session = tf.compat.v1.Session(config=config)
    # session = tf.Session(config=config)
    #
    # remove previously generated files or directories
    dirs_remove = ['__pycache__/', '~/.nv/']
    for dir_remove in dirs_remove:
        try:
            shutil.rmtree(dir_remove)
            print(f"{dir_remove} removed")
        except FileNotFoundError:
            print(f"{dir_remove} not found, continuing...")
            pass


def get_arg_best_trial(trials):
    """
    Returns index of best trial (trial at which loss in minimum).
    ----------
    ntrials: (int) number of trials used for hyperparameter optimization.
    """
    losses = [float(trial['result']['loss']) for trial in trials]
    arg_min_loss = np.argmin(losses)
    return arg_min_loss


def plot_predicted_phase_P(config, dct_mcd, data, sta, opath, plot_num, save_plot=True):
    """
    Creates plots for predicted P-phase time onsets.
    Two types of plots are created: i) entire predicted phase window, ii) zoom centered on refined phase pick, showing MCD results.
    ----------
    dct_mcd: (dict) dictionary containing MCD statistics of the predicted phase pick.
    dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
    data: (numpy array) array containing seismic stream amplitudes on which MCD is applied.
    sta: (str) station code of seismic stream.
    opath: (str) output path for created plots.
    plot_num: (int) index of processed phase onset, used for naming plots of predicted phase onsets.
    save_plot: (bool) True to save plots of predicted phase onset.
    """
    #
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    #
    opath_fig = f"{opath}/pick_plots"
    os.makedirs(opath_fig, exist_ok=True)
    #
    tpick_det = dct_mcd['pick']['tpick_det']
    tpick_pred = dct_mcd['pick']['tpick']
    tpick_pred_th1 = dct_mcd['pick']['tpick_th1']
    tpick_pred_th2 = dct_mcd['pick']['tpick_th2']
    terr_pre = dct_mcd['pick']['terr_pre']
    terr_pos = dct_mcd['pick']['terr_pos']
    pick_class = dct_mcd['pick']['pick_class']
    mc_pred = dct_mcd['mcd']['mc_pred']
    mc_pred_mean = dct_mcd['mcd']['mc_pred_mean']
    mc_pred_mean_arg_pick = dct_mcd['mcd']['mc_pred_mean_arg_pick']
    mc_pred_std_pick = dct_mcd['mcd']['mc_pred_std_pick']
    prob_th1 = dct_mcd['mcd']['prob_th1']
    prob_th2 = dct_mcd['mcd']['prob_th2']
    #
    # plot - phase window input for network
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.params['samp_freq']
    #
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=1.)
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='r', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='r', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"comp Z"
    # ax[-1].text(0.02, .95, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = 0.
    xmax = tr_win_x.max()
    ax[-1].set_xlim([xmin, xmax])
    ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .5))
    ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase P: {opath_fig}/{sta}_pred_P_{plot_num+1:02}.png")
    # print("#")
    ofig = f"{opath_fig}/{sta}_pred_P_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (zoom around predicted time pick)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2., zorder=1)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--', zorder=1)
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1., zorder=1)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
    ax_tmp.yaxis.set_minor_locator(ticker.MultipleLocator(.05))
    ax_tmp.axhline(mc_pred_mean[mc_pred_mean_arg_pick,0], c='magenta', lw=1., ls='--', zorder=2)
    ax_tmp.axhline(prob_th1, c='magenta', lw=1., ls='--', zorder=2)
    ax_tmp.axhline(prob_th2, c='magenta', lw=1., ls='--', zorder=2)
    #
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='r', lw=1.5, ls='-', clip_on=False, zorder=3)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='r', lw=1.5, ls='--', clip_on=False, zorder=3)
    ax[-1].vlines(x=tpick_pred_th1, ymin=-1., ymax=1., color='r', lw=1.5, ls=':', clip_on=False, zorder=3)
    ax[-1].vlines(x=tpick_pred_th2, ymin=-1., ymax=1., color='r', lw=1.5, ls=':', clip_on=False, zorder=3)
    # ax[-1].vlines(x=tpick_pred-tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].vlines(x=tpick_pred+tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # arg_pred = mc_pred_mean_arg_pick
    tr_label_1 = f"tpred = {tpick_pred:.3f}"
    tr_label_2 = f"terr(1 x pb_std) = (-{terr_pre:.3f}, +{terr_pos:.3f})"
    tr_label_3 = f"pick_class = {pick_class}"
    tr_label_4 = f"pb, pb_std = ({mc_pred_mean[mc_pred_mean_arg_pick,0]:.3f}, {mc_pred_std_pick:.3f})"
    # ax[-1].text(0.01, .975, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .935, tr_label_2, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .895, tr_label_3, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .855, tr_label_4, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = tpick_pred - .5
    xmax = tpick_pred + .5
    ax[-1].set_xlim([xmin, xmax])
    tick_major = np.arange(xmin, xmax + .1, .1)
    tick_minor = np.arange(xmin, xmax + .01, .02)
    # ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .1))
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    # ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.05))
    # ax[-1].xaxis.set_minor_locator(ticker.LinearLocator(5))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase P: {opath_fig}/{sta}_pred_P_mc_{plot_num+1:02}.png")
    print(tr_label_1)
    print(tr_label_2)
    print(tr_label_3)
    print(tr_label_4)
    # print("#")
    ofig = f"{opath_fig}/{sta}_pred_P_mc_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()


def plot_predicted_phase_S(config, dct_mcd, data, sta, opath, plot_num, save_plot=True):
    """
    Creates plots for predicted S-phase time onsets.
    Two types of plots are created: i) entire predicted phase window, ii) zoom centered on refined phase pick, showing MCD results.
    ----------
    dct_mcd: (dict) dictionary containing MCD statistics of the predicted phase pick.
    dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
    data: (numpy array) array containing seismic stream amplitudes on which MCD is applied.
    sta: (str) station code of seismic stream.
    opath: (str) output path for created plots.
    plot_num: (int) index of processed phase onset, used for naming plots of predicted phase onsets.
    save_plot: (bool) True to save plots of predicted phase onset.
    """
    #
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.width'] = 1.5
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 1.5
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    #
    opath_fig = f"{opath}/pick_plots"
    os.makedirs(opath_fig, exist_ok=True)
    #
    tpick_det = dct_mcd['pick']['tpick_det']
    tpick_pred = dct_mcd['pick']['tpick']
    tpick_pred_th1 = dct_mcd['pick']['tpick_th1']
    tpick_pred_th2 = dct_mcd['pick']['tpick_th2']
    terr_pre = dct_mcd['pick']['terr_pre']
    terr_pos = dct_mcd['pick']['terr_pos']
    pick_class = dct_mcd['pick']['pick_class']
    mc_pred = dct_mcd['mcd']['mc_pred']
    mc_pred_mean = dct_mcd['mcd']['mc_pred_mean']
    mc_pred_mean_arg_pick = dct_mcd['mcd']['mc_pred_mean_arg_pick']
    mc_pred_std_pick = dct_mcd['mcd']['mc_pred_std_pick']
    prob_th1 = dct_mcd['mcd']['prob_th1']
    prob_th2 = dct_mcd['mcd']['prob_th2']
    #
    # plot - phase window input for network (comp E)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.params['samp_freq']
    print(tr_win_x.shape, tr_win_y.shape)
    #
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=1.)
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"comp E"
    # ax[-1].text(0.02, .95, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = 0.
    xmax = tr_win_x.max()
    ax[-1].set_xlim([xmin, xmax])
    ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .5))
    ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_E_{plot_num+1:02}.png")
    # print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_E_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (comp N)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,1]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.params['samp_freq']
    # print(tr_win_x.shape, tr_win_y.shape)
    #
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=1.)
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    # tr_label_1 = f"comp N"
    # ax[-1].text(0.02, .95, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = 0.
    xmax = tr_win_x.max()
    ax[-1].set_xlim([xmin, xmax])
    ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .5))
    ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.1))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_N_{plot_num+1:02}.png")
    # print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_N_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (zoom around predicted time pick, comp E)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.params['samp_freq']
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2.)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--')
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1.)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
    ax_tmp.yaxis.set_minor_locator(ticker.MultipleLocator(.05))
    ax_tmp.axhline(mc_pred_mean[mc_pred_mean_arg_pick,0], c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th1, c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th2, c='magenta', lw=1., ls='--')
    #
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th1, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th2, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    # ax[-1].vlines(x=tpick_pred-tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].vlines(x=tpick_pred+tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # arg_pred = mc_pred_mean_arg_pick
    tr_label_1 = f"tpred = {tpick_pred:.3f}"
    tr_label_2 = f"terr(1 x pb_std) = (-{terr_pre:.3f}, +{terr_pos:.3f})"
    tr_label_3 = f"pick_class = {pick_class}"
    tr_label_4 = f"pb, pb_std = ({mc_pred_mean[mc_pred_mean_arg_pick,0]:.3f}, {mc_pred_std_pick:.3f})"
    # ax[-1].text(0.01, .975, tr_label_1, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .935, tr_label_2, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .895, tr_label_3, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.01, .855, tr_label_4, size=12., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = tpick_pred - .5
    xmax = tpick_pred + .5
    ax[-1].set_xlim([xmin, xmax])
    tick_major = np.arange(xmin, xmax + .1, .1)
    tick_minor = np.arange(xmin, xmax + .01, .02)
    # ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .1))
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    # ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.05))
    # ax[-1].xaxis.set_minor_locator(ticker.LinearLocator(5))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_E_mc_{plot_num+1:02}.png")
    ofig = f"{opath_fig}/{sta}_pred_S_E_mc_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for network (zoom around predicted time pick, comp N)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,1]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.params['samp_freq']
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2.)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--')
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1.)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
    ax_tmp.yaxis.set_minor_locator(ticker.MultipleLocator(.05))
    ax_tmp.axhline(mc_pred_mean[mc_pred_mean_arg_pick,0], c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th1, c='magenta', lw=1., ls='--')
    ax_tmp.axhline(prob_th2, c='magenta', lw=1., ls='--')
    #
    ax[-1].vlines(x=tpick_pred, ymin=-1.1, ymax=1., color='b', lw=1.5, ls='-', clip_on=False)
    ax[-1].vlines(x=tpick_det, ymin=-1., ymax=1.1, color='b', lw=1.5, ls='--', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th1, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    ax[-1].vlines(x=tpick_pred_th2, ymin=-1., ymax=1., color='b', lw=1.5, ls=':', clip_on=False)
    # ax[-1].vlines(x=tpick_pred-tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].vlines(x=tpick_pred+tpick_pred_std, ymin=-1., ymax=1.05, color='r', lw=1.5, ls='--', clip_on=False)
    # ax[-1].text(0.02, .975, tr_label_1, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.02, .935, tr_label_2, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.02, .895, tr_label_3, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    # ax[-1].text(0.02, .855, tr_label_4, size=10., ha='left', va='center', transform=ax[-1].transAxes)
    #
    xmin = tpick_pred - .5
    xmax = tpick_pred + .5
    ax[-1].set_xlim([xmin, xmax])
    tick_major = np.arange(xmin, xmax + .1, .1)
    tick_minor = np.arange(xmin, xmax + .01, .02)
    # ax[-1].xaxis.set_ticks(np.arange(xmin, xmax + .1, .1))
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    # ax[-1].xaxis.set_minor_locator(ticker.MultipleLocator(.05))
    # ax[-1].xaxis.set_minor_locator(ticker.LinearLocator(5))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel(f"time [s]")
    #
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_pred_S_N_mc_{plot_num+1:02}.png")
    print(tr_label_1)
    print(tr_label_2)
    print(tr_label_3)
    print(tr_label_4)
    # print("#")
    ofig = f"{opath_fig}/{sta}_pred_S_N_mc_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()


# def plot_predicted_wf_phases(best_params, dct_dets, dct_data, dct_param, dct_trigger, dct_picks, dct_out, dct_fmt, comps=['Z','E']):
def plot_predicted_wf_phases(config, data, model, comps=['Z','E']):
    """
    Plots predicted picks on seismic waveforms.
    ----------
    best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
    dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
    dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
    dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
    dct_picks: (dict) dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
    dct_out: (dict) dictionary defining DeepPhasePick output options.
    dct_fmt: (dict) dictionary of parameters defining some formatting for plotting prediction.
    comps: (list) seismic components to be plotted.
    """
    #
    # plot format parameters
    mpl.rcParams['xtick.major.size'] = 14
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 14
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 6
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['ytick.labelsize'] = 22
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['axes.labelsize'] = 22
    #
    # print("#")
    #
    # flag_data = dct_out['flag_data']
    best_params = model.model_detection['best_params']
    # stas_plot = list(dct_data[1]['st'].keys())
    # print(stas_plot)
    # print("#")
    #
    # print(list(dct_data.keys()))
    # for k in dct_data:
    #     print(dct_data[k]['twin'])
    #     print(dct_data[k]['st'][stas_plot[0]].__str__(extended=True))
    #
    # print("#")
    print("creating plots...")
    # for i in dct_data:
    for i in data.data:
        #
        for sta in data.data[i]['st']:
            #
            fig = plt.figure(figsize=(20., 5.*len(comps)))
            plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
            #
            for n, ch in enumerate(comps):
                #
                ax = []
                #
                # subplot - waveform trace (input for CNN)
                #
                tr = data.data[i]['st'][sta].select(channel='*'+ch)[0]
                dt = tr.stats.delta
                tr_y = tr.data
                # if dct_param['st_normalized']:
                #     y_max = np.array([np.abs(tr.data).max() for tr in dct_data[i]['st'][sta]]).max()
                #     tr_y /= y_max
                # print(tr_y.min(), tr_y.max())
                if config.params['st_normalized']:
                    y_max = np.abs(tr.data).max()
                    tr_y /= y_max
                tr_x = np.arange(tr.data.size) * dt
                # print(tr_y.min(), tr_y.max(), y_max)
                #
                # plot trace
                ax.append(fig.add_subplot(len(comps), 1, n+1))
                ax[-1].plot(tr_x, tr_y, c='gray', lw=.25)
                #
                tr_label = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
                ax[-1].text(0.02, .95, tr_label, size=14., ha='left', va='center', transform=ax[-1].transAxes)
                #
                # retrieve predicted P, S class probability time series
                #
                samp_dt = 1 / config.params['samp_freq']
                tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
                ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
                #
                x_prob_p = model.detections[i]['pred'][sta]['tt']+tp_shift
                y_prob_p = model.detections[i]['pred'][sta]['ts'][:,0]
                x_prob_s = model.detections[i]['pred'][sta]['tt']+ts_shift
                y_prob_s = model.detections[i]['pred'][sta]['ts'][:,1]
                #
                tstart_plot = tr.stats.starttime
                tend_plot = tr.stats.endtime
                # print("#")
                print(i, sta, ch, tstart_plot, tend_plot)
                #
                # lines at predicted picks
                #
                if sta in model.picks[i]:
                    #
                    for ii, k in enumerate(model.picks[i][sta]['P']['true_arg']):
                        #
                        # P pick corrected after phase picking
                        #
                        tstart_win = model.picks[i][sta]['P']['twd'][k]['tstart_win']
                        tend_win = model.picks[i][sta]['P']['twd'][k]['tend_win']
                        if config.picking['run_mcd']:
                            tpick_pred = model.picks[i][sta]['P']['twd'][k]['pick_ml']['tpick']
                            # tpick_th1 = model.picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_th1']
                            # tpick_th2 = model.picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_th2']
                            # pick_class = model.picks[i][sta]['P']['twd'][k]['pick_ml']['pick_class']
                        else:
                            tpick_pred = model.picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_det']
                        tp_plot = tstart_win - tstart_plot + tpick_pred
                        # ax[-1].axvline(tp_plot, c='r', lw=1.5, ls='--')
                        ax[-1].axvline(tp_plot, c='r', lw=1.5, ls='-')
                    #
                    for jj, l in enumerate(model.picks[i][sta]['S']['true_arg']):
                        #
                        # S pick corrected after phase picking
                        #
                        tstart_win = model.picks[i][sta]['S']['twd'][l]['tstart_win']
                        if config.picking['run_mcd']:
                            tpick_pred = model.picks[i][sta]['S']['twd'][l]['pick_ml']['tpick']
                            # tpick_th1 = model.picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_th1']
                            # tpick_th2 = model.picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_th2']
                            # pick_class = model.picks[i][sta]['S']['twd'][l]['pick_ml']['pick_class']
                        else:
                            tpick_pred = model.picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_det']
                        ts_plot = tstart_win - tstart_plot + tpick_pred
                        # ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='--')
                        ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='-')
                #
                # ax[-1].set_xlim([0, ...])
                ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,500.)[:])
                ylim = [-1., 1.]
                #
                # TODO: implement dct_fmt as part of config.data ??
                # try:
                #     ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,dct_fmt[flag_data][sta]['dx_tick'])[:])
                #     ylim = dct_fmt[flag_data][sta]['ylim1']
                # except:
                #     ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,dct_fmt[flag_data]['all']['dx_tick'])[:])
                #     ylim = dct_fmt[flag_data]['all']['ylim1']
                ax[-1].set_ylim(ylim)
            #
            plt.tight_layout()
            #
            opath = model.detections[i]['pred'][sta]['opath']
            tstr = opath.split('wf_')[1].split('/')[0]
            opath = f"{opath}/wf_plots"
            os.makedirs(opath, exist_ok=True)
            #
            # ofig = f"{opath}/plot_pred_{flag_data}_{sta}_{tstr}"
            ofig = f"{opath}/plot_pred_{config.data['net']}_{sta}_{tstr}"
            plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
            # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
            plt.close()


def plot_predicted_wf_phases_o(best_params, dct_dets, dct_data, dct_param, dct_trigger, dct_picks, dct_out, dct_fmt, comps=['Z','E']):
    """
    Plots predicted picks on seismic waveforms.
    ----------
    best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
    dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
    dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
    dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
    dct_picks: (dict) dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
    dct_out: (dict) dictionary defining DeepPhasePick output options.
    dct_fmt: (dict) dictionary of parameters defining some formatting for plotting prediction.
    comps: (list) seismic components to be plotted.
    """
    #
    # plot format parameters
    mpl.rcParams['xtick.major.size'] = 14
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 14
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 6
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['ytick.labelsize'] = 22
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['axes.labelsize'] = 22
    #
    print("#")
    #
    flag_data = dct_out['flag_data']
    stas_plot = list(dct_data[1]['st'].keys())
    print(stas_plot)
    print("#")
    #
    print(list(dct_data.keys()))
    for k in dct_data:
        print(dct_data[k]['twin'])
        print(dct_data[k]['st'][stas_plot[0]].__str__(extended=True))
    #
    print("#")
    print("creating plots...")
    for sta in stas_plot:
        #
        for i in dct_data:
            #
            fig = plt.figure(figsize=(20., 5.*len(comps)))
            plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
            #
            for n, ch in enumerate(comps):
                #
                ax = []
                #
                # subplot - waveform trace (input for CNN)
                #
                tr = dct_data[i]['st'][sta].select(channel='*'+ch)[0]
                dt = tr.stats.delta
                tr_y = tr.data
                # if dct_param['st_normalized']:
                #     y_max = np.array([np.abs(tr.data).max() for tr in dct_data[i]['st'][sta]]).max()
                #     tr_y /= y_max
                print(tr_y.min(), tr_y.max())
                if dct_param['st_normalized']:
                    y_max = np.abs(tr.data).max()
                    tr_y /= y_max
                tr_x = np.arange(tr.data.size) * dt
                print(tr_y.min(), tr_y.max(), y_max)
                #
                # plot trace
                ax.append(fig.add_subplot(len(comps), 1, n+1))
                ax[-1].plot(tr_x, tr_y, c='gray', lw=.25)
                #
                tr_label = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
                ax[-1].text(0.02, .95, tr_label, size=14., ha='left', va='center', transform=ax[-1].transAxes)
                #
                # retrieve predicted P, S class probability time series
                #
                tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * dct_param['samp_dt']
                ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * dct_param['samp_dt']
                #
                x_prob_p = dct_dets[i]['pred'][sta]['tt']+tp_shift
                y_prob_p = dct_dets[i]['pred'][sta]['ts'][:,0]
                x_prob_s = dct_dets[i]['pred'][sta]['tt']+ts_shift
                y_prob_s = dct_dets[i]['pred'][sta]['ts'][:,1]
                #
                tstart_plot = tr.stats.starttime
                tend_plot = tr.stats.endtime
                print("#")
                print(sta, i, tstart_plot, tend_plot)
                #
                # lines at predicted picks
                #
                if sta in dct_picks[i]:
                    #
                    for ii, k in enumerate(dct_picks[i][sta]['P']['true_arg']):
                        #
                        # P pick corrected after phase picking
                        #
                        tstart_win = dct_picks[i][sta]['P']['twd'][k]['tstart_win']
                        tend_win = dct_picks[i][sta]['P']['twd'][k]['tend_win']
                        if dct_trigger['mcd']['run_mcd']:
                            tpick_pred = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick']
                            # tpick_th1 = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_th1']
                            # tpick_th2 = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_th2']
                            # pick_class = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['pick_class']
                        else:
                            tpick_pred = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_det']
                        tp_plot = tstart_win - tstart_plot + tpick_pred
                        # ax[-1].axvline(tp_plot, c='r', lw=1.5, ls='--')
                        ax[-1].axvline(tp_plot, c='r', lw=1.5, ls='-')
                    #
                    for jj, l in enumerate(dct_picks[i][sta]['S']['true_arg']):
                        #
                        # S pick corrected after phase picking
                        #
                        tstart_win = dct_picks[i][sta]['S']['twd'][l]['tstart_win']
                        if dct_trigger['mcd']['run_mcd']:
                            tpick_pred = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick']
                            # tpick_th1 = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_th1']
                            # tpick_th2 = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_th2']
                            # pick_class = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['pick_class']
                        else:
                            tpick_pred = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_det']
                        ts_plot = tstart_win - tstart_plot + tpick_pred
                        # ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='--')
                        ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='-')
                #
                # ax[-1].set_xlim([0, ...])
                # ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,500.)[:])
                ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,dct_fmt[flag_data][sta]['dx_tick'])[:])
                ylim = dct_fmt[flag_data][sta]['ylim1']
                ax[-1].set_ylim(ylim)
            #
            plt.tight_layout()
            #
            opath = dct_dets[i]['pred'][sta]['opath']
            tstr = opath.split('wf_')[1].split('/')[0]
            opath = f"{opath}/wf_plots"
            os.makedirs(opath, exist_ok=True)
            #
            ofig = f"{opath}/plot_pred_{flag_data}_{sta}_{tstr}"
            plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
            plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
            plt.close()


# TODO: integrate into plot_predicted_wf_phases()
# TODO: implement option to plot prob funcs for P, S, Noise classes. Like is done for comps
def plot_predicted_wf_phases_prob(best_params, dct_dets, dct_data, dct_param, dct_trigger, dct_picks, dct_out, dct_fmt, comps=['Z','E']):
    """
    Plots predicted picks and discrete probability time series on seismic waveforms.
    ----------
    best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
    dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
    dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
    dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
    dct_picks: (dict) dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
    dct_out: (dict) dictionary defining DeepPhasePick output options.
    dct_fmt: (dict) dictionary of parameters defining some formatting for plotting prediction.
    comps: (list) seismic components to be plotted.
    """
    #
    # plot format parameters
    mpl.rcParams['xtick.major.size'] = 14
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 6
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 14
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 6
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.labelsize'] = 22
    mpl.rcParams['ytick.labelsize'] = 22
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['axes.labelsize'] = 22
    #
    print("#")
    flag_data = dct_out['flag_data']
    stas_plot = list(dct_data[1]['st'].keys())
    print(stas_plot)
    print("#")
    #
    print(list(dct_data.keys()))
    for k in dct_data:
        print(dct_data[k]['twin'])
        print(dct_data[k]['st'][stas_plot[0]].__str__(extended=True))
    #
    print("#")
    print("creating plots...")
    for sta in stas_plot:
        #
        for i in dct_data:
            #
            fig = plt.figure(figsize=(20., 5.*len(comps)))
            plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
            #
            for n, ch in enumerate(comps):
                #
                ax = []
                #
                # subplot - waveform trace (input for CNN)
                #
                tr = dct_data[i]['st'][sta].select(channel='*'+ch)[0]
                dt = tr.stats.delta
                tr_y = tr.data
                if dct_param['st_normalized']:
                    y_max = np.array([np.abs(tr.data).max() for tr in dct_data[i]['st'][sta]]).max()
                    tr_y /= y_max
                tr_x = np.arange(tr.data.size) * dt
                #
                # plot trace
                ax.append(fig.add_subplot(len(comps), 1, n+1))
                ax[-1].plot(tr_x, tr_y, c='gray', lw=.25)
                #
                tr_label = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
                ax[-1].text(0.02, .95, tr_label, size=14., ha='left', va='center', transform=ax[-1].transAxes)
                #
                # plot predicted P, S class probability functions
                #
                ax_tmp = ax[-1].twinx()
                tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * dct_param['samp_dt']
                ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * dct_param['samp_dt']
                #
                x_prob_p = dct_dets[i]['pred'][sta]['tt']+tp_shift
                y_prob_p = dct_dets[i]['pred'][sta]['ts'][:,0]
                x_prob_s = dct_dets[i]['pred'][sta]['tt']+ts_shift
                y_prob_s = dct_dets[i]['pred'][sta]['ts'][:,1]
                ax_tmp.plot(x_prob_p, y_prob_p, c='red', lw=0.75)
                ax_tmp.plot(x_prob_s, y_prob_s, c='blue', lw=0.75)
                # ax_tmp.plot(x_prob_p, y_prob_p, 'ro', ms=1.)
                # ax_tmp.plot(x_prob_s, y_prob_s, 'bo', ms=1.)
                #
                tstart_plot = tr.stats.starttime
                tend_plot = tr.stats.endtime
                print("#")
                print(sta, i, tstart_plot, tend_plot)
                #
                # lines at predicted picks
                #
                #
                if sta in dct_picks[i]:
                    #
                    for ii, k in enumerate(dct_picks[i][sta]['P']['true_arg']):
                        #
                        # P pick corrected after phase picking
                        #
                        tstart_win = dct_picks[i][sta]['P']['twd'][k]['tstart_win']
                        tend_win = dct_picks[i][sta]['P']['twd'][k]['tend_win']
                        if dct_trigger['mcd']['run_mcd']:
                            tpick_pred = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick']
                            # tpick_th1 = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_th1']
                            # tpick_th2 = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_th2']
                            # pick_class = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['pick_class']
                        else:
                            tpick_pred = dct_picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_det']
                        tp_plot = tstart_win - tstart_plot + tpick_pred
                        ax[-1].axvline(tp_plot, c='r', lw=1.5, ls='--')
                    #
                    for jj, l in enumerate(dct_picks[i][sta]['S']['true_arg']):
                        #
                        # S pick corrected after phase picking
                        #
                        tstart_win = dct_picks[i][sta]['S']['twd'][l]['tstart_win']
                        if dct_trigger['mcd']['run_mcd']:
                            tpick_pred = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick']
                            # tpick_th1 = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_th1']
                            # tpick_th2 = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_th2']
                            # pick_class = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['pick_class']
                        else:
                            tpick_pred = dct_picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_det']
                        ts_plot = tstart_win - tstart_plot + tpick_pred
                        ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='--')
                #
                # axes properties
                #
                ax_tmp.set_ylim([-0.05, 1.05])
                ax_tmp.yaxis.set_ticks(np.arange(0.,1.1,.1)[:])
                #
                # ax[-1].set_xlim([0, ...])
                # ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,500.)[:])
                ax[-1].xaxis.set_ticks(np.arange(0.,tr_x.max()+1.,dct_fmt[flag_data][sta]['dx_tick'])[:])
                ylim = dct_fmt[flag_data][sta]['ylim1']
                ax[-1].set_ylim(ylim)
            #
            plt.tight_layout()
            #
            opath = dct_dets[i]['pred'][sta]['opath']
            tstr = opath.split('wf_')[1].split('/')[0]
            opath = f"{opath}/wf_plots"
            os.makedirs(opath, exist_ok=True)
            #
            ofig = f"{opath}/plot_pred_{flag_data}_{sta}_{tstr}_prob"
            plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
            plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
            plt.close()


# TODO: implement option to plot prob funcs for P, S, Noise classes. Like is done for comps
# def plotly_predicted_wf_phases(best_params, dct_dets, dct_data, dct_param, dct_trigger, dct_picks, dct_out, comps=['Z','E']):
def plotly_predicted_wf_phases(config, data, model, comps=['Z','E']):
    """
    Creates interactive plot of predicted picks on seismic waveforms and discrete probability time series.
    Requires plotly library.
    ----------
    best_params: (dict) dictionary of best performing hyperparameters optimized for phase detection.
    dct_dets: (dict) dictionary containing predicted discrete phase class probability time series and preliminary phase picks.
    dct_param: (dict) dictionary of parameters defining how waveform data is to be preprocessed.
    dct_trigger: (dict) dictionary of parameters defining how predicted probability time series are used to obtain preliminary and refined phase onsets.
    dct_picks: (dict) dictionary containing preliminary (from detection stage) and refined (by Monte Carlo Dropout MCD in picking stage) phase picks.
    dct_out: (dict) dictionary defining DeepPhasePick output options.
    comps: (list) seismic components to be plotted.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import chart_studio.plotly as py
    import plotly.tools as tls
    from plotly.offline import init_notebook_mode, iplot
    import plotly.offline as pyoff
    #
    best_params = model.model_detection['best_params']
    # flag_data = dct_out['flag_data']
    nwin = len(data.data.keys())
    win_ks = list(data.data.keys())[:]
    #
    # print("#")
    for w in win_ks:
        #
        twin = [str(t) for t in data.data[w]['twin']]
        stas = sorted(data.data[w]['st'].keys())
        nstas = len(stas)
        print(f"plotting prediction on waveforms ({nstas} stations), window {w:02} / {nwin:02}: {twin}")
        #
        nstas_in_max = 5
        stas_arg = np.arange(0, nstas, nstas_in_max)
        for ns_arg, s_arg in enumerate(stas_arg):
            stas_in = stas[s_arg:s_arg+nstas_in_max]
            nstas_in = len(stas_in)
            print(stas_in)
            #
            fig = make_subplots(rows=nstas_in*1, cols=1,
                                # specs = [[{}, {}], [{}, {}], [{}, {}]],
                                # specs = [[{"secondary_y": False}], [{"secondary_y": True}]] * nstas_in,
                                # specs = [[{"secondary_y": True}], [{"secondary_y": True}]] * nstas_in,
                                specs = [[{"secondary_y": True}]] * nstas_in,
                                horizontal_spacing = 0.025,
                                vertical_spacing = 0.01
                            )
            #
            nrow = 0
            nxax, nyax = 0, 0
            shapes, annotations = [], []
            dct_layout = {}
            #
            stas_plot = []
            for sta in stas_in:
                #
                nrow += 1
                stas_plot.append(sta)
                chs = comps
                #
                nxax += 1
                nyax += 1
                for ch in chs:
                    #
                    # add subplot - processed waveform (input for CNN)
                    #
                    tr = data.data[w]['st'][sta].select(channel='*'+ch)[0]
                    dt = tr.stats.delta
                    tr_y = tr.data
                    if config.params['st_normalized']:
                        y_max = np.array([np.abs(tr.data).max() for tr in data.data[w]['st'][sta]]).max()
                        tr_y /= y_max
                    tr_x = np.arange(tr.data.size) * dt
                    dct_layout[nyax] = {
                        'xmin': tr_x.min(), 'xmax': tr_x.max(), 'ymin': tr_y.min(), 'ymax': tr_y.max(),
                        'anchor': [nxax, nyax], 'label': 'pred', 'tr_label': f"{tr.stats.network}.{tr.stats.station}",
                    }
                    #
                    # plot trace
                    trace_name = f"trace{ch} {sta}"
                    visible_flag = True
                    line_color_flag = "#606060"
                    if ch in ["N", "E"]:
                        visible_flag = "legendonly"
                        line_color_flag = "#a0a0a0"
                    fig.add_trace(
                        go.Scatter(x=tr_x, y=tr_y, mode='lines', line={'color':line_color_flag, 'width':1.}, name=trace_name, visible=visible_flag), row=nrow, col=1,
                    )
                #
                # plot predicted P,S class probability functions
                #
                samp_dt = 1 / config.params['samp_freq']
                tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
                ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
                #
                nyax += 1
                stas_plot.append(sta)
                dct_layout[nyax] = {
                    'xmin': tr_x.min(), 'xmax': tr_x.max(), 'ymin': 0, 'ymax': 1.,
                    'anchor': [nxax, nyax], 'label': 'prob',
                }
                fig.add_trace(
                    go.Scatter(x=model.detections[w]['pred'][sta]['tt']+tp_shift, y=model.detections[w]['pred'][sta]['ts'][:,0], mode='lines', line={'color':'red', 'width':.75},
                        name='prob P'), row=nrow, col=1, secondary_y=True,
                    # go.Scatter(x=model.detections[w]['pred'][sta]['tt']+tp_shift, y=model.detections[w]['pred'][sta]['ts'][:,0], mode='lines+markers', line={'color':'red', 'width':.75},
                    #     marker={'color':'red', 'size':2}, name='prob P'), row=nrow, col=1, secondary_y=True,
                )
                fig.add_trace(
                    go.Scatter(x=model.detections[w]['pred'][sta]['tt']+ts_shift, y=model.detections[w]['pred'][sta]['ts'][:,1], mode='lines', line={'color':'blue', 'width':.75},
                        name='prob S'), row=nrow, col=1, secondary_y=True,
                    # go.Scatter(x=model.detections[w]['pred'][sta]['tt']+ts_shift, y=model.detections[w]['pred'][sta]['ts'][:,1], mode='lines+markers', line={'color':'blue', 'width':.75},
                    #     marker={'color':'blue', 'size':2}, name='prob S'), row=nrow, col=1, secondary_y=True,
                )
            #
            # x-axes properties
            #
            # print(fig['layout'])
            # print("#")
            # for i in dct_layout:
            #     print(i, dct_layout[i])
            #
            for i in range(1, nxax+1):
                if i == 1:
                    fig['layout'][f"xaxis"].update(dtick=30., autorange=True)
                else:
                    fig['layout'][f"xaxis{i}"].update(dtick=30., autorange=True)
                #
                if i == nxax:
                    title_str = f"time [s], window = {twin[0]} -- {twin[1]}"
                    # fig['layout'][f"xaxis{i}"].update(title='time [s]')
                    fig['layout'][f"xaxis{i}"].update(title=title_str)
            #
            # y-axes properties
            #
            for i in range(1, nyax+1):
                if dct_layout[i]['label'] == 'pred':
                    if config.params['st_normalized']:
                        if i == 1:
                            fig['layout'][f"yaxis"].update(range=[-1., 1.], autorange=False)
                        else:
                            fig['layout'][f"yaxis{i}"].update(range=[-1., 1.], autorange=False)
                    else:
                        dy = abs(dct_layout[i]['ymax'] - dct_layout[i]['ymin'])
                        ymin = dct_layout[i]['ymin'] - dy*.05
                        ymax = dct_layout[i]['ymax'] + dy*.05
                        if i == 1:
                            fig['layout'][f"yaxis"].update(range=[ymin, ymax], autorange=False)
                        else:
                            fig['layout'][f"yaxis{i}"].update(range=[ymin, ymax], autorange=False)
                else:
                    dy = abs(dct_layout[i]['ymax'] - dct_layout[i]['ymin'])
                    ymin = dct_layout[i]['ymin'] - dy*.05
                    ymax = dct_layout[i]['ymax'] + dy*.05
                    fig['layout'][f"yaxis{i}"].update(range=[ymin, ymax], autorange=False)
            #
            # print("")
            # print(fig['layout'])
            #
            # shapes
            #
            # lines at predicted picks
            #
            for i in range(1, nyax+1):
                if dct_layout[i]['label'] != 'pred':
                    continue
                #
                sta_plot = stas_plot[i-1]
                if i == 1:
                    x_ref, y_ref = "x", "y"
                    y_0, y_1 = fig['layout']["yaxis"]['range']
                else:
                    xr, yr = dct_layout[i]['anchor']
                    x_ref, y_ref = f"x{xr}", f"y{yr}"
                    y_0, y_1 = fig['layout'][f"yaxis{yr}"]['range']
                #
                tstart_plot = data.data[w]['twin'][0]
                #
                if sta_plot in model.picks[w]:
                    # print(model.picks[w][sta_plot]['P']['true_arg'])
                    for ii, k in enumerate(model.picks[w][sta_plot]['P']['true_arg']):
                        #
                        # P pick corrected after phase picking
                        #
                        tstart_win = model.picks[w][sta_plot]['P']['twd'][k]['tstart_win']
                        tend_win = model.picks[w][sta_plot]['P']['twd'][k]['tend_win']
                        if config.picking['run_mcd']:
                            #
                            # plot refined picks by DPP RNN model (MCD applied on detection window)
                            # -> beware: this would produce a shift between plotted highest probability value and pick time
                            #
                            tpick_pred = model.picks[w][sta_plot]['P']['twd'][k]['pick_ml']['tpick']
                            # tpick_th1 = model.picks[w][sta_plot]['P']['twd'][k]['pick_ml']['tpick_th1']
                            # tpick_th2 = model.picks[w][sta_plot]['P']['twd'][k]['pick_ml']['tpick_th2']
                            # pick_class = model.picks[w][sta_plot]['P']['twd'][k]['pick_ml']['pick_class']
                        else:
                            #
                            # plot detected picks by the DPP CNN model
                            #
                            tpick_pred = model.picks[w][sta_plot]['P']['twd'][k]['pick_ml']['tpick_det']
                        tp_plot = tstart_win - tstart_plot + tpick_pred
                        # print('P', tstart_win, tend_win, tpick_pred, tp_plot)
                        shapes.append(go.layout.Shape(
                            type="line",
                            x0=tp_plot,
                            y0=y_0,
                            x1=tp_plot,
                            y1=y_1,
                            xref=x_ref,
                            yref=y_ref,
                            line=dict(
                                color="red", width=1., dash="dash"
                            )))
                    #
                    for jj, l in enumerate(model.picks[w][sta_plot]['S']['true_arg']):
                        #
                        # S pick corrected after phase picking
                        #
                        tstart_win = model.picks[w][sta_plot]['S']['twd'][l]['tstart_win']
                        if config.picking['run_mcd']:
                            tpick_pred = model.picks[w][sta_plot]['S']['twd'][l]['pick_ml']['tpick']
                            # tpick_th1 = model.picks[w][sta_plot]['S']['twd'][l]['pick_ml']['tpick_th1']
                            # tpick_th2 = model.picks[w][sta_plot]['S']['twd'][l]['pick_ml']['tpick_th2']
                            # pick_class = model.picks[w][sta_plot]['S']['twd'][l]['pick_ml']['pick_class']
                        else:
                            tpick_pred = model.picks[w][sta_plot]['S']['twd'][l]['pick_ml']['tpick_det']
                        ts_plot = tstart_win - tstart_plot + tpick_pred
                        # print('S', tstart_win, tend_win, tpick_pred, ts_plot)
                        shapes.append(go.layout.Shape(
                            type="line",
                            x0=ts_plot,
                            y0=y_0,
                            x1=ts_plot,
                            y1=y_1,
                            xref=x_ref,
                            yref=y_ref,
                            line=dict(
                                color="blue", width=1., dash="dash"
                            )))
            #
            # annotations
            #
            for i in range(1, nyax+1):
                if dct_layout[i]['label'] in ['trr', 'prob']:
                    continue
                #
                if i == 1:
                    x_ref, y_ref = "x", "y"
                else:
                    xr, yr = dct_layout[i]['anchor']
                    x_ref, y_ref = f"x{xr}", f"y{yr}"
                #
                dy = abs(dct_layout[i]['ymax'] - dct_layout[i]['ymin'])
                dx = abs(dct_layout[i]['xmax'] - dct_layout[i]['xmin'])
                x_txt = dct_layout[i]['xmin'] + dx*.05
                #
                if dct_layout[i]['label'] == 'pred':
                    y_txt = -.85
                else:
                    y_txt = dct_layout[i]['ymin'] + dy*.025
                #
                tr_label = dct_layout[i]['tr_label']
                annotations.append(go.layout.Annotation(
                    x=x_txt, y=y_txt, xref=x_ref, yref=y_ref,
                    text=tr_label,
                    showarrow=False,
                    font={"size": 14, "color": "black"},
                    # width=20.,
                    # height=20.
                ))
            #
            fig.update_layout(
                # height=200*nstas_in, width=1800, showlegend=True,
                height=300*nstas_in, width=1800,
                margin = dict(l=0, r=0, b=0, t=0),
                showlegend=True,
                dragmode="zoom",
                shapes=shapes, annotations=annotations,
                # paper_bgcolor='rgba(0,0,0,0)',
                # plot_bgcolor='rgba(0,0,0,0)',
            )
            # fig.show()
            #
            stas = list(data.data[w]['st'].keys())
            opath = data.data[w]['opath']
            # opath = data.data[w]['st'][stas[0]]['opath']
            tstr = opath.split('wf_')[1].split('/')[0]
            opath = f"{opath}/wf_plots"
            os.makedirs(opath, exist_ok=True)
            # ofig = f"{opath}/plotly_pred_{flag_data}_{sta}_{tstr}_win_{w:02}_{(ns_arg+1):02}.html"
            ofig = f"{opath}/plotly_pred_{config.data['net']}_{tstr}_win_{w:02}_{(ns_arg+1):02}.html"
            #
            pyoff.offline.plot(fig, auto_open=False, filename=f"{ofig}")


