#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
script_dir = os.path.dirname(os.path.realpath(__file__))

def create_stim_rate_file(exc_rate, delay=1000):
    Lems = ET.Element("Lems")
    # include Inputs.xml
    include = ET.SubElement(Lems, "Include")
    include.set("file", "Inputs.xml")
    exc_spikegen = ET.SubElement(Lems, "spikeGeneratorDelay")
    exc_spikegen.set("id", "mossySpiker")
    exc_spikegen.set("period", "{} ms".format(1000./exc_rate))
    exc_spikegen.set("delay", "{} ms".format(delay))
    # save to disk
    with open(script_dir+"/inputRate.xml", "w") as freq_file:
        ET.ElementTree(Lems).write(freq_file)

def moving_causal_average(signal, window_length, timestep):
    window_size = int(round(window_length/timestep))
    window = np.hstack((np.ones(window_size), np.zeros(window_size)))/float(window_size)
    return np.convolve(signal, window, 'same')

def main():
    #exc_rate_range = np.array([1,2,5,10,20,30,40,50,60,70,80,90,100], dtype=np.float)
    exc_rate_range = np.array([50])
    window_size_range = np.linspace(6, 300, 10)
    averaged_voltage = np.zeros((exc_rate_range.size, window_size_range.size))
    sim_duration = 3000. # ms
    timestep = 0.05 # ms

    out_filename = "voltage.dat"

    for k, exc_rate in enumerate(exc_rate_range):
        create_stim_rate_file(exc_rate)
        if os.path.isfile(out_filename):
            os.remove(out_filename)
        proc = subprocess.Popen(["jnml "+script_dir+"/integrationWindow.xml"],
                                shell=True,
                                stdout=subprocess.PIPE)
        proc.communicate()
        
        sim_data = np.loadtxt(out_filename)
        timepoints = sim_data[:,0]
        voltage = sim_data[:,1]
        start_timepoint = np.searchsorted(timepoints, 0.5)
        stop_timepoint = np.searchsorted(timepoints, 2.5)
        averaged_voltage = np.array([moving_causal_average(voltage, length, timestep)[start_timepoint:stop_timepoint] for length in window_size_range])

        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(averaged_voltage.transpose())
        ax[1].plot(sim_data[:,1])
        #ax.imshow(averaged_voltage, aspect='equal')
    plt.show()

if __name__ == "__main__":
    main()


