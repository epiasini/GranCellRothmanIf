import os
import sys
import numpy as np
from matplotlib import pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(script_dir))

import simulation_tools

def main():
    exc_rate = 50
    sim_duration = 1
    n_stims = 4
    sim_data = simulation_tools.simulate_poisson_stimulation(exc_rate, sim_duration, n_stims)
    time = sim_data[:,0] * 1000
    voltage = sim_data[:,1] * 1000
    derivative = np.diff(voltage)
    voltage[derivative<-1] = 40
    
    fig, ax = plt.subplots()
    ax.plot(time, voltage, linewidth=2.5, color='k')
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("membrane potential (mV)")
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.show()
    

if __name__ == "__main__":
    main()
