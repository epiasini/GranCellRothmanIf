#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.rc('font', family='Helvetica', size=20)
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))

import simulation_tools

def main():
    # jason's f-f curve for model cell #156 in Schwartz2012
    # (supposedly the one closest to having average parameter values),
    # in presence of tonic GABA
    jason_stim_range = np.arange(60, 660, 60)/4
    jason_rates_156 = np.array([0.0, 0.7755102040816326, 4.55813953488372, 20.65625, 49.46153846153846, 88.9047619047619, 139.2222222222222, 187.6875, 227.64285714285717, 253.7142857142857])

    scale_exc = False
    scale_inh = True

    n_stims_range = np.arange(1,9,1)
    exc_rate_range = np.arange(10, 155, 10)
    #exc_scaling_factor_range = np.arange(0.5, 1.6, 0.25)
    out_rates = np.zeros(shape=(n_stims_range.size, exc_rate_range.size))
    sim_duration = 6 # s
    
    for i, n_stims in enumerate(n_stims_range):
        if scale_exc:
            exc_scaling_factor = 4./n_stims
            inh_scaling_factor = 1.
        elif scale_inh:
            exc_scaling_factor = 1.
            inh_scaling_factor = n_stims/4.
        else:
            exc_scaling_factor = 1.
            inh_scaling_factor = 1.
        for k, exc_rate in enumerate(exc_rate_range):
            sim_data = simulation_tools.simulate_poisson_stimulation([exc_rate for each in range(n_stims)], sim_duration, inh_scaling_factor=inh_scaling_factor, exc_scaling_factor=exc_scaling_factor)

            spike_count = sim_data[-1,2]
            out_rates[i,k] = float(spike_count) / sim_duration
            print("inh: {}; exc: {}; input: {}Hz; output: {:.2f}Hz".format(inh_scaling_factor, exc_scaling_factor, exc_rate, out_rates[i,k]))

    # plotting setup
    ff_fig, ff_ax = plt.subplots()
    #ff_ax.plot(jason_stim_range, jason_rates_156, color='k', linewidth=1.5, marker='o', label='Schwartz2012 cell #156')
    cm = plt.get_cmap('RdYlBu_r') 
    c_norm  = colors.Normalize(vmin=n_stims_range[0], vmax=n_stims_range[-1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    matplotlib.rcParams.update({'axes.labelsize': 30})
    matplotlib.rcParams.update({'xtick.labelsize': 30})
    matplotlib.rcParams.update({'ytick.labelsize': 30})

    for i, exc_scaling_factor in enumerate(n_stims_range):
        color = scalar_map.to_rgba(exc_scaling_factor)
        ff_ax.plot(exc_rate_range,
                   out_rates[i],
                   linewidth=1.5,
                   marker='o',
                   color=color)
        print('input: {}'.format(exc_rate_range))
        print('output: {}'.format(out_rates[i]))

    #ff_ax.legend(loc='best')
    ff_ax.set_xlabel("MF firing rate (Hz)")
    ff_ax.set_ylabel("GrC firing rate (Hz)")
    ff_ax.xaxis.set_ticks_position('bottom')
    ff_ax.yaxis.set_ticks_position('left')
    ff_ax.spines['top'].set_color('none')
    ff_ax.spines['right'].set_color('none')

    cb_ax, kw = matplotlib.colorbar.make_axes(ff_ax)
    cb = matplotlib.colorbar.ColorbarBase(cb_ax, cmap=cm, norm=c_norm, **kw)
    cb.set_ticks(n_stims_range)
    cb.set_label('dendrites')

    plt.show()

if __name__ == "__main__":
    main()


