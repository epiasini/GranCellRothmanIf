#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))

import simulation_tools

def main():
    exc_rate_range = np.array([50])
    out_rates = np.zeros_like(exc_rate_range)
    sim_duration = 60. # s
    n_bins = 200
    upper_hist_limit = 0.300 # s

    for k, exc_rate in enumerate(exc_rate_range):
        sim_data = simulation_tools.simulate_poisson_stimulation(exc_rate, sim_duration)
        
        timepoints = sim_data[:,0]
        voltage = sim_data[:,1]
        spike_times = timepoints[np.diff(sim_data[:,2]) != 0]
        out_rates[k] = float(spike_times.size) / sim_duration
        print("input: {}Hz; output: {}Hz".format(exc_rate, out_rates[k]))

        relative_times = np.zeros(shape=(spike_times.size, spike_times.size))
        for s, t in enumerate(spike_times):
            relative_times[s] = spike_times - t
        small_relative_times = relative_times[relative_times > 0]
        small_relative_times = small_relative_times[small_relative_times < upper_hist_limit]
        if small_relative_times.size:
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(small_relative_times, bins=n_bins, color='k', normed=True)
            uniform_baseline = 1. / (bins[-1] - bins[0])

            ax.plot([0, upper_hist_limit], [uniform_baseline, uniform_baseline], color='r', linewidth=1.5)
            ax.set_title('Spike time autocorrelation for stim rate {}Hz'.format(exc_rate))
            ax.set_xlabel('lag (s)')
            ax.set_ylabel('relative frequency (spike probability density)')

    if out_rates.size > 1:
        ff_fig, ff_ax = plt.subplots()
        ff_ax.plot(exc_rate_range, out_rates, color='k', linewidth=1.5)
        ff_ax.set_xlabel("MF firing rate (Hz)")
        ff_ax.set_xlabel("GrC firing rate (Hz)")
        plt.show()

if __name__ == "__main__":
    main()


