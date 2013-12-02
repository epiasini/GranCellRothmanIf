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
    # jason's f-f curve for model cell #156 in Schwartz2012
    # (supposedly the one closest to having average parameter values),
    # in presence of tonic GABA
    jason_stim_range = np.arange(60, 660, 60)/4
    jason_rates_156 = np.array([0.0, 0.7755102040816326, 4.55813953488372, 20.65625, 49.46153846153846, 88.9047619047619, 139.2222222222222, 187.6875, 227.64285714285717, 253.7142857142857])

    exc_rate_range = np.arange(10, 155, 5)
    out_rates = np.zeros_like(exc_rate_range)
    sim_duration = 60. # s

    for k, exc_rate in enumerate(exc_rate_range):
        sim_data = simulation_tools.simulate_poisson_stimulation(exc_rate, sim_duration)

        spike_count = sim_data[-1,2]
        out_rates[k] = float(spike_count) / sim_duration
        print("input: {}Hz; output: {}Hz".format(exc_rate, out_rates[k]))

    ff_fig, ff_ax = plt.subplots()
    ff_ax.plot(jason_stim_range, jason_rates_156, color='k', linewidth=1.5, marker='o', label='Schwartz2012 cell #156')
    ff_ax.plot(exc_rate_range, out_rates, color='r', linewidth=1.5, marker='o', label='2013 (LEMS)')
    ff_ax.set_title('GrC model: rate I/O with 4 Poisson stimuli and tonic GABA')
    ff_ax.legend(loc='best')
    ff_ax.set_xlabel("MF firing rate (Hz)")
    ff_ax.set_ylabel("GrC firing rate (Hz)")
    plt.show()

if __name__ == "__main__":
    main()


