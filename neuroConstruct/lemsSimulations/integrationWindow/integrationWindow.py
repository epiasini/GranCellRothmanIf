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
    exc_rate_range = np.array([80])
    out_rates = np.zeros_like(exc_rate_range)
    sim_duration = 400. # s
    n_bins = 200
    upper_hist_limit = 0.100 # s

    for k, exc_rate in enumerate(exc_rate_range):
        exc_rates = [exc_rate, exc_rate, exc_rate, 10]
        sim_data = simulation_tools.simulate_poisson_stimulation(exc_rates, sim_duration)
        
        timepoints = sim_data[:,0]
        voltage = sim_data[:,1]
        spike_times = timepoints[np.diff(sim_data[:,2]) != 0]
        out_rates[k] = float(spike_times.size) / sim_duration
        print("input: {}Hz; output: {}Hz".format(exc_rate, out_rates[k]))

        relative_times = []
        for s, t in enumerate(spike_times):
            stop_idx = np.searchsorted(spike_times, t+upper_hist_limit)
            relative_times.extend((spike_times[s+1:stop_idx] - t).tolist())
        small_relative_times = np.array(relative_times)
        if small_relative_times.size:
            fig, ax = plt.subplots()
            bin_length = upper_hist_limit/n_bins
            weights = np.ones_like(small_relative_times)/(bin_length*sim_duration)
            n, bins, patches = ax.hist(small_relative_times*1000, bins=n_bins, color='k', weights=weights, normed=False)
            uniform_baseline_spikes_per_bin = out_rates[k]*bin_length
            ax.plot([0, bins[-1]], [out_rates[k]**2, out_rates[k]**2], color='r', linewidth=1.5)
            ax.set_title('Spike time autocorrelation for stim rate {}Hz'.format(exc_rate))
            ax.set_xlabel('lag (ms)')
            ax.set_ylabel('autocorrelation (Hz$^2$)')

    if out_rates.size > 1:
        ff_fig, ff_ax = plt.subplots()
        ff_ax.plot(exc_rate_range, out_rates, color='k', linewidth=1.5)
        ff_ax.set_xlabel("MF firing rate (Hz)")
        ff_ax.set_xlabel("GrC firing rate (Hz)")
    plt.show()

if __name__ == "__main__":
    main()


