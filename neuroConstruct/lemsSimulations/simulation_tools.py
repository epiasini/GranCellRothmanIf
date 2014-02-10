import os
import subprocess
import datetime
import xml.dom.minidom
import numpy as np
import h5py
import tempfile
import shutil
from os.path import dirname, abspath, join
from xml.etree import ElementTree as ET
from matplotlib import pyplot as plt


def pretty_xml_string(element):
    # sometimes jLEMS freaks out on unprettified xml. This is meant to
    # work around that.
    ugly_xml = xml.dom.minidom.parseString(ET.tostring(element))
    return ugly_xml.toprettyxml()

class StimulusLEMSAdapter(object):
    def __init__(self, exc_rates):
        self.lems = ET.Element("Lems")
        for k, rate in enumerate(exc_rates):
            exc_spikegen = ET.SubElement(self.lems, "spikeGeneratorRefPoisson")
            exc_spikegen.set("id", "mossySpiker{}".format(k))
            exc_spikegen.set("averageRate", "{} Hz".format(rate))
            exc_spikegen.set("minimumISI", "1 ms")

class SimulationConfigurationLEMSAdapter(object):
    def __init__(self,
                 length,
                 n_stims,
                 out_filename,
                 step=0.05,
                 target="poissonStimNetwork",
                 save_synaptic_conductances=False):
        self.lems = ET.Element("Lems")
        simulation = ET.SubElement(self.lems, "Simulation")
        simulation.set("id", "generatedSimulation")
        simulation.set("length", "{} s".format(length))
        simulation.set("step", "{} ms".format(step))
        simulation.set("target", target)
        output_file = ET.SubElement(simulation, "OutputFile")
        output_file.set("id", "of0")
        output_file.set("fileName", out_filename)
        voltage_column = ET.SubElement(output_file, "OutputColumn")
        voltage_column.set("id", "v")
        voltage_column.set("quantity", "GrCPop[0]/v")
        spike_column = ET.SubElement(output_file, "OutputColumn")
        spike_column.set("id", "sc")
        spike_column.set("quantity", "SpikeRecorderPop[0]/spikeRecorder/spikeCount")
        if save_synaptic_conductances:
            for stim in range(n_stims):
                for conn_type in ['AMPA', 'NMDA']:
                    syn_column = ET.SubElement(output_file, "OutputColumn")
                    syn_column.set("id", "{}{}".format(conn_type, stim))
                    syn_column.set("quantity", "GrCPop[0]/RothmanMFToGrC{}{}/g".format(conn_type, stim))

class NetworkLEMSAdapter(object):
    def __init__(self, n_stims):
        self.lems = ET.Element("Lems")
        network = ET.SubElement(self.lems, "network")
        network.set("id", "poissonStimNetwork")
        grc_pop = ET.SubElement(network, "population")
        grc_pop.set("id", "GrCPop")
        grc_pop.set("component", "IaF_GrC_no_tonic_GABA")
        grc_pop.set("size", "1")
        sr_pop = ET.SubElement(network, "population")
        sr_pop.set("id", "SpikeRecorderPop")
        sr_pop.set("component", "IaF_GrC_no_tonic_GABA")
        sr_pop.set("size", "1")
        for mf in range(n_stims):
            mf_pop = ET.SubElement(network, "population")
            mf_pop.set("id", "mossySpikerPop{}".format(mf))
            mf_pop.set("component", "mossySpiker{}".format(mf))
            mf_pop.set("size", "1")
            for conn_type in ['AMPA', 'NMDA']:
                conn = ET.SubElement(network, "synapticConnection")
                conn.set("from", "mossySpikerPop{}[0]".format(mf))
                conn.set("to", "GrCPop[0]")
                conn.set("synapse", "RothmanMFToGrC{}{}".format(conn_type, mf))
                conn.set("destination", "synapses")
        sr_conn = ET.SubElement(network, "synapticConnection")
        sr_conn.set("from", "GrCPop[0]")
        sr_conn.set("to", "SpikeRecorderPop[0]")
        sr_conn.set("synapse", "spikeRecorder")
        sr_conn.set("destination", "synapses")

class CustomLEMSInclude(object):
    def __init__(self, filename):
        self.lems = ET.Element("Lems")
        include_element = ET.SubElement(self.lems, "Include")
        include_element.set("file", filename)
        
def set_tonic_GABA(project_dir, conductance_in_nS, reversal_potential=-79.1):
    """
    Modify the IaF_GrC_no_tonic_GABA.nml file in the working copy of
    the nC project to change the level of tonic GABA from what is
    originally set in the model.

    """
    ET.register_namespace('', 'http://www.neuroml.org/schema/neuroml2')
    filename = project_dir + "/cellMechanisms/IaF_GrC_no_tonic_GABA/IaF_GrC_no_tonic_GABA.nml"
    tree = ET.parse(filename)
    root = tree.getroot()
    cell = root.find("{http://www.neuroml.org/schema/neuroml2}iafRefCell")
    old_conductance = float(cell.get('leakConductance').rstrip('nS'))
    old_reversal_potential = float(cell.get('leakReversal').rstrip('mV'))
    new_conductance = old_conductance + conductance_in_nS
    new_reversal_potential = (old_conductance * old_reversal_potential + conductance_in_nS * reversal_potential)/new_conductance
    cell.set('leakConductance', '{}nS'.format(new_conductance))
    cell.set('leakReversal', '{}mV'.format(new_reversal_potential))
    tree.write(filename, encoding="UTF-8", xml_declaration=True)

def scale_synaptic_conductance(project_dir, component_name, scaling_factor, attributes_to_scale, units='nS'):
    ET.register_namespace('', 'http://www.neuroml.org/schema/neuroml2')
    filename = project_dir + "/cellMechanisms/{0}/{0}.nml".format(component_name)
    tree = ET.parse(filename)
    root = tree.getroot()
    synapse = root.findall(".//*[@id='{}']".format(component_name))[0]
    for attribute in attributes_to_scale:
        old_conductance = float(synapse.get(attribute).rstrip(units))
        new_conductance = old_conductance * scaling_factor
        synapse.set(attribute, '{}{}'.format(new_conductance, units))
    tree.write(filename, encoding="UTF-8", xml_declaration=True)


def scale_excitatory_conductances(project_dir, scaling_factor):
    """
    Modify the RothmanMFToGrCAMPA.nml and RothmanMFToGrCNMDA.nml files
    in the working copy of of the nC project to scale the total amount
    of excitatory conductance.

    """
    # scale AMPA
    scale_synaptic_conductance(project_dir=project_dir,
                               component_name='RothmanMFToGrCAMPA',
                               scaling_factor=scaling_factor,
                               attributes_to_scale=['directAmp1',
                                                    'directAmp2',
                                                    'spilloverAmp1',
                                                    'spilloverAmp2',
                                                    'spilloverAmp3'])
    # scale NMDA
    scale_synaptic_conductance(project_dir=project_dir,
                               component_name='RothmanMFToGrCNMDA',
                               scaling_factor=scaling_factor,
                               attributes_to_scale=['directAmp1',
                                                    'directAmp2'])


def simulate_poisson_stimulation(exc_rates, sim_duration_in_s, save_synaptic_conductances=False, inh_scaling_factor=1., exc_scaling_factor=1.):
    n_stims = len(exc_rates)
    # create file where simulation data will be stored
    sim_data_file = tempfile.NamedTemporaryFile(delete=False, dir='./')
    sim_data_file.close()

    project_dir = dirname(dirname(abspath(__file__)))
    cell_mechs_dir = join(project_dir, 'cellMechanisms')

    # make temporary copy of mech files
    temp_project_dir = tempfile.mkdtemp()
    temp_cell_mechs_dir = join(temp_project_dir, 'cellMechanisms')
    shutil.copytree(cell_mechs_dir, temp_cell_mechs_dir)

    # scale conductances
    gGABA_base_in_nS = 0.438
    total_GABA_conductance_in_nS = gGABA_base_in_nS*inh_scaling_factor
    set_tonic_GABA(temp_project_dir, total_GABA_conductance_in_nS)
    scale_excitatory_conductances(temp_project_dir, exc_scaling_factor)

    # load base template for xml simulation description
    ET.register_namespace('', 'http://www.neuroml.org/schema/neuroml2')
    template_filename = join(join(project_dir, "lemsSimulations"), "poisson_inputs_simulation_template.xml")
    lems_tree = ET.parse(template_filename)
    lems_root = lems_tree.getroot()

    # define includes for custom component types and components
    custom_lems_defs_dir = join(project_dir, "lemsDefinitions")
    include_filenames = [join(custom_lems_defs_dir, 'spikeRecorder.xml'),
                         join(custom_lems_defs_dir, 'spikeGeneratorRefPoisson.xml')]
    for component_name in ['RothmanMFToGrCAMPA', 'RothmanMFToGrCNMDA', 'IaF_GrC_no_tonic_GABA']:
        include_filenames.append(join(join(temp_cell_mechs_dir, component_name), component_name+'.nml'))
    lems_includes = [CustomLEMSInclude(filename).lems for filename in include_filenames]




    # define replicas of synaptic components
    synaptic_replicas = []
    for stim in range(n_stims):
        for syn_name in ['RothmanMFToGrCAMPA', 'RothmanMFToGrCNMDA']:
            tree = ET.parse(join(join(temp_cell_mechs_dir, syn_name), syn_name+'.nml'))
            root = tree.getroot()
            element = root.findall("./*[@id='{}']".format(syn_name))[0]
            element.set("id", "{}{}".format(syn_name, stim))
            synaptic_replicas.append(element)

    # define procedurally generated lems elements
    lems_stim = StimulusLEMSAdapter(exc_rates).lems
    lems_net = NetworkLEMSAdapter(n_stims).lems
    lems_sim = SimulationConfigurationLEMSAdapter(length=sim_duration_in_s,
                                                  n_stims=n_stims,
                                                  out_filename=sim_data_file.name,
                                                  save_synaptic_conductances=save_synaptic_conductances).lems
    
    # insert custom includes in simulation description template
    position_includes = 5
    position_syn_replicas = 6
    position_stim = 7
    position_net = 8
    position_sim = 9
    for k, inc in enumerate(lems_includes):
        lems_root.insert(position_includes, inc[0])
        position_syn_replicas += 1
        position_stim += 1
        position_net += 1
        position_sim += 1
    # insert synaptic replicas
    for replica in synaptic_replicas:
        lems_root.insert(position_syn_replicas, replica)
        position_stim += 1
        position_net += 1
        position_sim += 1
    # insert procedurally generated lems
    for stim in range(n_stims):
        lems_root.insert(position_stim, lems_stim[stim])
        position_net += 1
        position_sim += 1
    lems_root.insert(position_net, lems_net[0])
    lems_root.insert(position_sim, lems_sim[0])
    # write simulation description to disk
    sim_description_file = tempfile.NamedTemporaryFile(delete=False, dir='./')
    sim_description_file.write(pretty_xml_string(lems_root))
    sim_description_file.close()
    # run jLEMS
    proc = subprocess.Popen(["jnml {}".format(sim_description_file.name)],
                            shell=True,
                            stdout=subprocess.PIPE)
    proc.communicate()
    # remove simulation description file
    os.remove(sim_description_file.name)
    # remove temp mech dir
    shutil.rmtree(temp_project_dir)
    # read in output firing rate and remove simulation data file
    out_data = np.loadtxt(sim_data_file.name)
    os.remove(sim_data_file.name)
    return out_data
