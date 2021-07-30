# coding: utf-8

"""
This module contains additional useful functions used by DeepPhasePick method.

Author: Hugo Soto Parada (October, 2020)
Contact: soto@gfz-potsdam.de, hugosotoparada@gmail.com

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import re, sys, os, shutil, gc
import pickle


def export_dict2pckl(dct, opath):
    """
    Exports dictionary as pickle file.

    Parameters
    ----------
    dct: dict
        Input dictionary.
    opath: str
        Output path to export pickle file.
    """
    with open(opath, 'wb') as pout:
        pickle.dump(dct, pout)


def import_pckl2dict(ipath):
    """
    Imports pickle file to dictionary and returns this dictionary.

    Parameters
    ----------
    ipath: str
        Path to pickle file.
    """
    with open(ipath, 'rb') as pin:
        dct = pickle.load(pin)
    return dct


def init_session():
    """
    Sets up tensorflow v2.x / keras session.
    """
    #
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
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

    Parameters
    ----------
    trials: list
        List of hyperopt trials results in hyperparameter optimization.

    Returns
    -------
    arg_min_loss: int
        Index corresponding to best trial (trial at which loss in minimum) in trials object.
    """
    losses = [float(trial['result']['loss']) for trial in trials]
    arg_min_loss = np.argmin(losses)
    return arg_min_loss


def plot_predicted_phase_P(config, dct_mcd, data, sta, opath, plot_num):
    """
    Creates plots for predicted P-phase time onsets.
    Two types of plots are created, showing:
    i) refined phase pick in picking window, ii) zoom centered on refined phase pick and Monte Carlo Dropout (MCD) results.

    Parameters
    ----------
    config: instance of config.Config
        Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
    dct_mcd: dict
        Dictionary containing MCD statistics of the predicted phase pick.
    data: ndarray
        3D array containing seismic stream amplitudes on which MCD is applied.
    sta: str
        Station code of seismic stream.
    opath: str
        Output path for saving figure of predicted phase onsets.
    plot_num: int
        Index of processed phase onset, used for figure names of predicted phase onsets.
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
    # plot - phase window input for RNN
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    # print('plot_predicted_phase_P', data.shape, data.ndim)
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.data_params['samp_freq']
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
    ax[-1].set_xlabel(f"Time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase P: {opath_fig}/{sta}_P_{plot_num+1:02}.png")
    ofig = f"{opath_fig}/{sta}_P_Z_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for RNN (zoom around predicted time pick and MCD results)
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
    ax_tmp.set_ylabel("Probability")
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
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel("Time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase P: {opath_fig}/{sta}_P_mc_{plot_num+1:02}.png")
    print(tr_label_1)
    print(tr_label_2)
    print(tr_label_3)
    print(tr_label_4)
    ofig = f"{opath_fig}/{sta}_P_Z_mcd_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()


def plot_predicted_phase_S(config, dct_mcd, data, sta, opath, plot_num):
    """
    Creates plots for predicted S-phase time onsets.
    Two types of plots are created, showing:
    i) refined phase pick in picking window, ii) zoom centered on refined phase pick and Monte Carlo Dropout (MCD) results.

    Parameters
    ----------
    config: instance of config.Config
        Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
    dct_mcd: dict
        Dictionary containing MCD statistics of the predicted phase pick.
    data: ndarray
        3D array containing seismic stream amplitudes on which MCD is applied.
    sta: str
        Station code of seismic stream.
    opath: str
        Output path for saving figure of predicted phase onsets.
    plot_num: int
        Index of processed phase onset, used for figure names of predicted phase onsets.
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
    # plot - phase window input for RNN (comp E)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    # print('plot_predicted_phase_S', data.shape, data.ndim)
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.data_params['samp_freq']
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
    ax[-1].set_xlabel("Time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_S_E_{plot_num+1:02}.png")
    ofig = f"{opath_fig}/{sta}_S_E_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for RNN (comp N)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,1]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.data_params['samp_freq']
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
    ax[-1].set_xlabel("Time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_S_N_{plot_num+1:02}.png")
    ofig = f"{opath_fig}/{sta}_S_N_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for RNN (zoom around predicted time pick and MCD results, comp E)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,0]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.data_params['samp_freq']
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2.)
    # ax[-1].plot(tr_win_x, tr_win_y, c='k', marker='o', ms=.5)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--')
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1.)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.set_ylabel("Probability")
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
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel("Time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_S_E_mc_{plot_num+1:02}.png")
    ofig = f"{opath_fig}/{sta}_S_E_mcd_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()
    #
    # plot - phase window input for RNN (zoom around predicted time pick and MCD results, comp N)
    #
    fig = plt.figure(figsize=(7*1, 3*1))
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    #
    # plot trace + label
    tr_win_y = data[0,:,1]
    tr_win_x = np.arange(tr_win_y.shape[0]) / config.data_params['samp_freq']
    ax[-1].plot(tr_win_x, tr_win_y, c='gray', lw=2.)
    #
    # plot output binary probs
    ax_tmp = ax[-1].twinx()
    for l in range(len(mc_pred)):
        ax_tmp.plot(tr_win_x, mc_pred[l,:,0], c='magenta', lw=.2, ls='--')
    ax_tmp.plot(tr_win_x, mc_pred_mean[:,0], c='magenta', lw=1.)
    ax_tmp.set_ylim([0., 1.])
    ax_tmp.set_ylabel("Probability")
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
    ax[-1].xaxis.set_major_locator(ticker.FixedLocator(tick_major))
    ax[-1].xaxis.set_minor_locator(ticker.FixedLocator(tick_minor))
    ax[-1].set_ylim([-1., 1.])
    ax[-1].set_xlabel("Time [s]")
    #
    plt.tight_layout()
    print(f"plotting predicted phase S: {opath_fig}/{sta}_S_N_mc_{plot_num+1:02}.png")
    print(tr_label_1)
    print(tr_label_2)
    print(tr_label_3)
    print(tr_label_4)
    ofig = f"{opath_fig}/{sta}_S_N_mcd_{plot_num+1:02}"
    plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
    # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
    plt.close()


def plot_predicted_phases(config, data, model, plot_comps=['Z','E'], plot_probs=[], shift_probs=True):
    """
    Plots predicted P- and S-phase picks on seismic waveforms and additionally predicted discrete class probability time series.

    Parameters
    ----------
    config: instance of config.Config
        Contains user configuration of seismic waveform data and how this data is processed in DeepPhasePick.
    data: instance of data.Data
        Contains selected seismic waveform data on which phase detection is applied.
    model: instance of model.Model
        Contains best models and relevant results obtained from hyperparameter optimization for phase detection and picking.
    plot_comps: list of str, optional
        Seismic components to be plotted. It can be any of vertical ('Z'), east ('E'), and north ('N').
        By default vertical and east components are plotted.
    plot_probs: list of str, optional
        Discrete class probability time series to be plotted. It can be any of 'P', 'S' and 'N' (Noise) classes.
        By default no probability time series are plotted.
    shift_probs: bool, optional.
        If True (default), plotted probability time series are shifted in time according to the optimized hyperparameters defining the picking window for each class.
        See Figure S1 in Soto and Schurr (2020).
    """
    #
    # plot format parameters
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 16
    #
    best_params = model.model_detection['best_params']
    add_rows = 0
    if len(plot_probs) > 0:
        add_rows += 1
    #
    print("creating plots...")
    for i in data.data:
        #
        for sta in data.data[i]['st']:
            #
            fig = plt.figure(figsize=(12., 4*(len(plot_comps)+add_rows)))
            plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0)
            #
            for n, ch in enumerate(plot_comps):
                #
                ax = []
                #
                # subplot - waveform trace (input for CNN)
                #
                tr = data.data[i]['st'][sta].select(channel='*'+ch)[0]
                dt = tr.stats.delta
                tr_y = tr.data
                y_max = np.abs(tr.data).max()
                tr_y /= y_max
                tr_x = np.arange(tr.data.size) * dt
                # print(tr_y.min(), tr_y.max(), y_max)
                #
                # plot trace
                ax.append(fig.add_subplot(len(plot_comps)+add_rows, 1, n+1))
                ax[-1].plot(tr_x, tr_y, c='gray', lw=.2)
                # ax[-1].plot(tr_x, tr_y, c='k', lw=.2)
                #
                # retrieve predicted P, S class probability time series
                #
                samp_dt = 1 / config.data_params['samp_freq']
                if shift_probs:
                    tp_shift = (best_params['frac_dsamp_p1']-.5) * best_params['win_size'] * samp_dt
                    ts_shift = (best_params['frac_dsamp_s1']-.5) * best_params['win_size'] * samp_dt
                    tn_shift = (best_params['frac_dsamp_n1']-.5) * best_params['win_size'] * samp_dt
                else:
                    tp_shift = 0
                    ts_shift = 0
                    tn_shift = 0
                #
                # plot trace label
                # tr_label = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
                tr_label = f"{tr.stats.channel}"
                box_label = dict(boxstyle='square', facecolor='white', alpha=.9)
                ax[-1].text(0.02, .95, tr_label, size=14., ha='left', va='center', transform=ax[-1].transAxes, bbox=box_label)
                #
                tstart_plot = tr.stats.starttime
                tend_plot = tr.stats.endtime
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
                            # tpick_pred = model.picks[i][sta]['P']['twd'][k]['pick_ml']['tpick_det']
                            tpick_pred = model.picks[i][sta]['P']['twd'][k]['pick_ml_det']
                        tp_plot = tstart_win - tstart_plot + tpick_pred
                        if ii == 0:
                            ax[-1].axvline(tp_plot, c='r', lw=1.5, ls='-', label='P pick')
                        else:
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
                            # tpick_pred = model.picks[i][sta]['S']['twd'][l]['pick_ml']['tpick_det']
                            tpick_pred = model.picks[i][sta]['S']['twd'][l]['pick_ml_det']
                        ts_plot = tstart_win - tstart_plot + tpick_pred
                        if jj == 0:
                            ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='-', label='S pick')
                        else:
                            ax[-1].axvline(ts_plot, c='b', lw=1.5, ls='-')
                #
                ylim = [-1., 1.]
                ax[-1].set_ylim(ylim)
                ax[-1].set_xlim([0, tend_plot - tstart_plot])
                if n == len(plot_comps)-1:
                    plt.legend(loc='lower left', fontsize=14.)
                    if add_rows == 0:
                        ax[-1].set_xlabel("Time [s]")
            #
            # plot predicted P, S, Noise class probability functions
            #
            if len(plot_probs) > 0:
                ax.append(fig.add_subplot(len(plot_comps)+add_rows, 1, len(plot_comps)+1))
                ax[-1].set_xlim([0, tend_plot - tstart_plot])
                ax[-1].set_xlabel("Time [s]")
                ax[-1].set_ylim([-.05, 1.05])
                ax[-1].set_ylabel("Probability")
            if 'P' in plot_probs:
                x_prob_p = model.detections[i][sta]['tt']+tp_shift
                y_prob_p = model.detections[i][sta]['ts'][:,0]
                ax[-1].plot(x_prob_p, y_prob_p, c='red', lw=0.75, label='P')
            if 'S' in plot_probs:
                x_prob_s = model.detections[i][sta]['tt']+ts_shift
                y_prob_s = model.detections[i][sta]['ts'][:,1]
                ax[-1].plot(x_prob_s, y_prob_s, c='blue', lw=0.75, label='S')
            if 'N' in plot_probs:
                x_prob_n = model.detections[i][sta]['tt']+tn_shift
                y_prob_n = model.detections[i][sta]['ts'][:,2]
                ax[-1].plot(x_prob_n, y_prob_n, c='k', lw=0.75, label='N')
            if len(plot_probs) > 0:
                plt.legend(loc='lower left', fontsize=14.)
            #
            plt.tight_layout()
            #
            opath = model.detections[i][sta]['opath']
            tstr_start = tr.stats.starttime.strftime("%Y%m%dT%H%M%S")
            tstr_end = tr.stats.endtime.strftime("%Y%m%dT%H%M%S")
            opath = f"{opath}/wf_plots"
            os.makedirs(opath, exist_ok=True)
            #
            ofig = f"{opath}/{config.data['net']}_{sta}_{tstr_start}_{tstr_end}"
            plt.savefig(f"{ofig}.png", bbox_inches='tight', dpi=90)
            # plt.savefig(f"{ofig}.eps", format='eps', bbox_inches='tight', dpi=150)
            plt.close()
