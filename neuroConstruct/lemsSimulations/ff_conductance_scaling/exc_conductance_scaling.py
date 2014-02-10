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

    n_stims = 4
    exc_rate_range = np.arange(10, 155, 10)
    exc_scaling_factor_range = np.arange(0.5, 1.6, 0.25)
    out_rates = np.zeros(shape=(exc_scaling_factor_range.size, exc_rate_range.size))
    sim_duration = 6 # s
    
    for i, exc_scaling_factor in enumerate(exc_scaling_factor_range):
        for k, exc_rate in enumerate(exc_rate_range):
            sim_data = simulation_tools.simulate_poisson_stimulation([exc_rate for each in range(n_stims)], sim_duration, inh_scaling_factor=1, exc_scaling_factor=exc_scaling_factor)

            spike_count = sim_data[-1,2]
            out_rates[i,k] = float(spike_count) / sim_duration
            print("exc: {}; input: {}Hz; output: {:.2f}Hz".format(exc_scaling_factor,
                                                                  exc_rate,
                                                                  out_rates[i,k]))

    # plotting setup
    ff_fig, ff_ax = plt.subplots()
    #ff_ax.plot(jason_stim_range, jason_rates_156, color='k', linewidth=1.5, marker='o', label='Schwartz2012 cell #156')
    cm = plt.get_cmap('RdYlBu_r') 
    c_norm  = colors.Normalize(vmin=exc_scaling_factor_range[0], vmax=exc_scaling_factor_range[-1])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    matplotlib.rcParams.update({'axes.labelsize': 30})
    matplotlib.rcParams.update({'xtick.labelsize': 30})
    matplotlib.rcParams.update({'ytick.labelsize': 30})

    for i, exc_scaling_factor in enumerate(exc_scaling_factor_range):
        color = scalar_map.to_rgba(exc_scaling_factor)
        ff_ax.plot(exc_rate_range,
                   out_rates[i],
                   linewidth=1.5,
                   marker='o',
                   color=color)
        print('input: {}'.format(exc_rate_range))
        print('output: {}'.format(out_rates[i]))

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


# 2 dendrites, double excitation:
# input: [ 10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95
#  100 105 110 115 120 125 130 135 140 145 150]
# output: [   0.28333333    0.8           2.08333333    4.36666667    9.13333333
#    15.8          24.43333333   30.71666667   41.46666667   56.45
#    72.06666667   88.1         105.83333333  119.86666667  136.55        149.15
#   172.56666667  181.9         196.1         208.98333333  218.1         226.1
#   238.36666667  244.71666667  253.03333333  261.53333333  268.86666667
#   274.11666667  280.55      ]

# 8 dendrites, half-sized excitatory synapses:
# input: [ 10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95
#  100 105 110 115 120 125 130 135 140 145 150]
# output: [  0.00000000e+00   0.00000000e+00   0.00000000e+00   3.33333333e-02
#    1.16666667e-01   4.33333333e-01   1.73333333e+00   5.75000000e+00
#    1.33833333e+01   2.72333333e+01   4.72000000e+01   6.44000000e+01
#    9.25333333e+01   1.16833333e+02   1.44450000e+02   1.67433333e+02
#    1.86850000e+02   2.03766667e+02   2.19200000e+02   2.30166667e+02
#    2.39416667e+02   2.49666667e+02   2.57966667e+02   2.63533333e+02
#    2.70233333e+02   2.75766667e+02   2.81666667e+02   2.85466667e+02
#    2.89883333e+02]

# 8 dendrites, double tonic GABA:
# input: [ 10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95
# 100 105 110 115 120 125 130 135 140 145 150]
# output: [  0.00000000e+00   3.83333333e-01   2.58333333e+00   1.18500000e+01
#    3.30000000e+01   6.68000000e+01   1.10166667e+02   1.54200000e+02
#    2.02683333e+02   2.38966667e+02   2.60016667e+02   2.82866667e+02
#    3.01316667e+02   3.14750000e+02   3.24383333e+02   3.32433333e+02
#    3.42200000e+02   3.48050000e+02   3.53433333e+02   3.58400000e+02
#    3.63233333e+02   3.67183333e+02   3.70966667e+02   3.74516667e+02
#    3.77433333e+02   3.80400000e+02   3.82666667e+02   3.85250000e+02
#    3.87516667e+02]

