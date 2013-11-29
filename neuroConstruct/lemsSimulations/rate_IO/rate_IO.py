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
    exc_rate_range = np.arange(10, 155, 5)
    out_rates = np.zeros_like(exc_rate_range)
    sim_duration = 60. # s

    for k, exc_rate in enumerate(exc_rate_range):
        sim_data = simulation_tools.simulate_poisson_stimulation(exc_rate, sim_duration)

        spike_count = sim_data[-1,2]
        out_rates[k] = float(spike_count) / sim_duration
        print("input: {}Hz; output: {}Hz".format(exc_rate, out_rates[k]))

    ff_fig, ff_ax = plt.subplots()
    ff_ax.plot(exc_rate_range, out_rates, color='k', linewidth=1.5, marker='o')
    ff_ax.set_xlabel("MF firing rate (Hz)")
    ff_ax.set_xlabel("GrC firing rate (Hz)")
    plt.show()

if __name__ == "__main__":
    main()


