#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import numpy as np
import matplotlib
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

    n_stims_range = np.arange(1, 11, 1, dtype=np.int)
    n_stims_range = np.array([4])
    exc_rate_range = np.arange(10, 155, 5)
    out_rates = np.zeros(shape=(n_stims_range.size, exc_rate_range.size))
    sim_duration = 30 # s
    
    for d, n_stims in enumerate(n_stims_range):
        for k, exc_rate in enumerate(exc_rate_range):
            sim_data = simulation_tools.simulate_poisson_stimulation(exc_rate, sim_duration, n_stims)

            spike_count = sim_data[-1,2]
            out_rates[d,k] = float(spike_count) / sim_duration
            print("stims: {}; input: {}Hz; output: {:.2f}Hz".format(n_stims,
                                                                    exc_rate,
                                                                    out_rates[d,k]))

    # plotting setup
    ff_fig, ff_ax = plt.subplots()
    #ff_ax.plot(jason_stim_range, jason_rates_156, color='k', linewidth=1.5, marker='o', label='Schwartz2012 cell #156')
    cm = plt.get_cmap('RdYlBu') 
    c_norm  = colors.Normalize(vmin=0, vmax=n_stims_range[-1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    matplotlib.rcParams.update({'axes.labelsize': 30})
    matplotlib.rcParams.update({'xtick.labelsize': 30})
    matplotlib.rcParams.update({'ytick.labelsize': 30})

    for d, n_stims in enumerate(n_stims_range):
        color = scalar_map.to_rgba(n_stims)
        ff_ax.plot(exc_rate_range,
                   out_rates[d],
                   linewidth=1.5,
                   marker='o',
                   color='r')

    ff_ax.set_title('GrC model: rate I/O with {} to {} Poisson stimuli and tonic GABA'.format(n_stims_range[0], n_stims_range[-1]))
    ff_ax.legend(loc='best')
    ff_ax.set_xlabel("MF firing rate (Hz)")
    ff_ax.set_ylabel("GrC firing rate (Hz)")
    ff_ax.xaxis.set_ticks_position('bottom')
    ff_ax.yaxis.set_ticks_position('left')
    ff_ax.spines['top'].set_color('none')
    ff_ax.spines['right'].set_color('none')

    plt.show()

if __name__ == "__main__":
    main()


