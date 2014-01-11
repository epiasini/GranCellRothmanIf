import os
import sys
import numpy as np
from matplotlib import pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))

import simulation_tools

def main():
    exc_rate = 80
    sim_duration = 1
    n_stims = 3
    sim_data = simulation_tools.simulate_poisson_stimulation(exc_rate, sim_duration, n_stims, save_synaptic_conductances=True)
    time = sim_data[:,0] * 1000
    voltage = sim_data[:,1] * 1000
    derivative = np.diff(voltage)
    voltage[derivative<-1] = 40
    
    fig_v, ax_v = plt.subplots()
    ax_v.plot(time, voltage, linewidth=2.5, color='k')
    ax_v.set_xlabel("time (ms)")
    ax_v.set_ylabel("membrane potential (mV)")
    ax_v.xaxis.set_ticks_position('bottom')
    ax_v.yaxis.set_ticks_position('left')
    ax_v.spines['top'].set_color('none')
    ax_v.spines['right'].set_color('none')

    fig_c, ax_c = plt.subplots(nrows=n_stims, ncols=1, sharex=True)
    for stim in range(n_stims):
        conductance_AMPA = sim_data[:,3+2*stim]
        conductance_NMDA = sim_data[:,4+2*stim]
        ax_c[stim].plot(time, conductance_AMPA, linewidth=2.5, color='r')
        ax_c[stim].plot(time, conductance_NMDA, linewidth=2.5, color='#5F04B4')

    plt.show()
    

if __name__ == "__main__":
    main()
