#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
script_dir = os.path.dirname(os.path.realpath(__file__))

def create_stim_rate_file(exc_rate, delay=1000, stim_type='regular'):
    Lems = ET.Element("Lems")
    # include Inputs.xml
    include = ET.SubElement(Lems, "Include")
    include.set("file", "Inputs.xml")
    if stim_type=='regular':
        exc_spikegen = ET.SubElement(Lems, "spikeGeneratorDelay")
        exc_spikegen.set("period", "{} ms".format(1000./exc_rate))
        exc_spikegen.set("delay", "{} ms".format(delay))
    elif stim_type=='poisson':
        exc_spikegen = ET.SubElement(Lems, "spikeGeneratorPoisson")
        exc_spikegen.set("averageRate", "{} Hz".format(exc_rate))
    exc_spikegen.set("id", "mossySpiker")
    # save to disk
    with open(script_dir+"/inputRate.xml", "w") as freq_file:
        ET.ElementTree(Lems).write(freq_file)

def moving_causal_average(signal, window_length, timestep):
    window_size = int(round(window_length/timestep))
    window = np.hstack((np.ones(window_size), np.zeros(window_size)))/float(window_size)
    return np.convolve(signal, window, 'same')

def main():
    exc_rate_range = np.array([40])
    sim_duration = 50. # s
    n_bins = 200
    upper_hist_limit = 0.300

    out_filename = "voltage.dat"

    for k, exc_rate in enumerate(exc_rate_range):
        create_stim_rate_file(exc_rate, stim_type='poisson')
        if os.path.isfile(out_filename):
            os.remove(out_filename)
        proc = subprocess.Popen(["jnml "+script_dir+"/integrationWindow.xml"],
                                shell=True,
                                stdout=subprocess.PIPE)
        proc.communicate()
        
        sim_data = np.loadtxt(out_filename)
        timepoints = sim_data[:,0]
        voltage = sim_data[:,1]
        spike_times = timepoints[np.diff(sim_data[:,2]) != 0]
        if not spike_times.size:
            print('no output spikes for input firing rate of {}Hz!'.format(exc_rate))
            break

        relative_times = np.zeros(shape=(spike_times.size, spike_times.size))
        for s, t in enumerate(spike_times):
            relative_times[s] = spike_times - t
        #relative_times = (np.atleast_2d(spike_times) - np.atleast_2d(spike_times).transpose()).flatten()
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
    plt.show()

if __name__ == "__main__":
    main()


