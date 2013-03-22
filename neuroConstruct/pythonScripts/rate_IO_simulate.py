"""
Simulates the granule cell model with 4 mossy fibre stimuli, each
firing at the given frequency and returns the corresponding output
firing frequency.
"""
import os
import random
import time
import sys
import shutil
from java.lang import System, Long
from java.io import File

from ucl.physiol.neuroconstruct.project import ProjectManager
from ucl.physiol.neuroconstruct.neuron import NeuronFileManager
from ucl.physiol.neuroconstruct.nmodleditor.processes import ProcessManager
from ucl.physiol.neuroconstruct.cell.utils import CellTopologyHelper
from ucl.physiol.neuroconstruct.utils import NumberGenerator

timestamp = str(time.time())
pm = ProjectManager(None, None)
project_path = '../GranCellRothmanIF.ncx'
project_file = File(project_path)
project = pm.loadProject(project_file)

sim_config_name = 'synapse_test'

sim_config = project.simConfigInfo.getSimConfig(sim_config_name)
sim_duration = 100000
sim_config.setSimDuration(sim_duration)
project.neuronSettings.setNoConsole()

input_rate = float(sys.argv[1])

# generate
nC_seed = 1234
pm.doGenerate(sim_config_name, nC_seed)
while pm.isGenerating():
    time.sleep(0.001)
print('network generated')

sim_ref = 'io_rate_' + timestamp + '_' + str(input_rate)
sim_path = '../simulations/' + sim_ref
project.simulationParameters.setReference(sim_ref)
# set stim input_rate
input_rate_in_kHz = input_rate/1000.
stim = project.elecInputInfo.getStim('relay_stim')
stim.setRate(NumberGenerator(input_rate_in_kHz))
project.elecInputInfo.updateStim(stim)
# generate and compile neuron files
print "Generating NEURON scripts..."
simulator_seed = random.getrandbits(32)
project.neuronFileManager.generateTheNeuronFiles(sim_config, None, NeuronFileManager.RUN_HOC,simulator_seed)
compile_process = ProcessManager(project.neuronFileManager.getMainHocFile())
compile_success = compile_process.compileFileWithNeuron(0,0)
# simulate
if compile_success:
    print "Simulating " + sim_ref
    pm.doRunNeuron(sim_config)
    timefile_path = sim_path + '/time.dat'
    while not os.path.exists(timefile_path):
        time.sleep(1)

# calculate output firing rate
out_file_path = sim_path + '/GrCs_0.SPIKE_min40.spike'

out_file = open(out_file_path)
n_spikes = -1
for k, spiketime in enumerate(out_file):
    n_spikes+=1
out_file.close()
shutil.rmtree(sim_path)

print("Output firing rate is %f" % (1000 * float(n_spikes)/sim_duration))

System.exit(0)
