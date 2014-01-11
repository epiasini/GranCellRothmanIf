import os
import sys
import numpy as np
from matplotlib import pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))

import simulation_tools

def bottom_left_axes(ax):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

def main():
    exc_rates = [80, 80, 10, 10] # Hertz
    sim_duration = 0.25 # seconds

    linewidth = 2.5
    fontsize = 16

    n_stims = len(exc_rates)
    sim_data = simulation_tools.simulate_poisson_stimulation(exc_rates, sim_duration, save_synaptic_conductances=True)
    time = sim_data[:,0] * 1000
    voltage = sim_data[:,1] * 1000
    derivative = np.diff(voltage)
    voltage[derivative<-1] = 40
    
    # plot GrC membrane potential
    fig_v, ax_v = plt.subplots()
    ax_v.plot(time, voltage, linewidth=linewidth, color='b')
    ax_v.set_xlabel("time (ms)", fontsize=fontsize)
    ax_v.set_ylabel("membrane potential (mV)", fontsize=fontsize)
    bottom_left_axes(ax_v)

    # plot excitatory conductance trains
    fig_c, ax_c = plt.subplots(nrows=n_stims+1, ncols=1, sharex=True, sharey=True)
    for stim in range(n_stims):
        conductance_AMPA = sim_data[:,3+2*stim]*1e9
        conductance_NMDA = sim_data[:,4+2*stim]*1e9
        ax_c[stim].plot(time, conductance_AMPA, linewidth=linewidth, color='r')
        ax_c[stim].plot(time, conductance_NMDA, linewidth=linewidth, color='#5F04B4')
    # plot tonic inhibition
    ax_c[-1].plot(time, 0.438+np.zeros_like(time), linewidth=linewidth, color='g')

    # prettify conductance axes
    for ax in ax_c:
        ax.axes.get_yaxis().set_ticks([0, 0.3, 0.6])
        bottom_left_axes(ax)
    ax_c[-1].set_xlabel("time (ms)", fontsize=fontsize)
    ax_c[2].set_ylabel("conductance (nS)", fontsize=fontsize)

    plt.show()
    

if __name__ == "__main__":
    main()
