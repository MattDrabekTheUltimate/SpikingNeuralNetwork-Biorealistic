# SpikingNeuralNetwork-Biorealistic
This repository implements a spiking neural network (SNN) simulation using the Izhikevich neuron model. It incorporates biorealistic features to explore learning and information processing in neural networks

Neuromorphic Simulation

This project simulates a neuromorphic network using the Lava framework. The simulation includes dynamic baseline adjustments, neuromodulation, synaptic plasticity, and real-time visualization.

Features

Dynamic Baseline Adjustments: Utilizes an Ornstein-Uhlenbeck process to generate time-varying baselines.
Neuromodulation: Incorporates dopamine, serotonin, and norepinephrine effects on neuron excitability.
Synaptic Plasticity: Implements spike-timing-dependent plasticity (STDP) for synaptic weight updates.
Real-Time Visualization: Provides real-time plotting of neuron activities.
Parallel Processing: Uses parallel processing for efficient synaptic weight updates.
Installation

Prerequisites
Python 3.8 or higher
pip (Python package installer)


Steps

Clone the repository:

git clone https: adress of this repo
cd neuromorphic-simulation
Install the required packages:


pip install -r requirements.txt
Usage

Running the Simulation
To run the simulation, execute the simulation.py script:



python simulation.py
Simulation Parameters
You can adjust the simulation parameters directly in the simulation.py file. Key parameters include:

num_neurons: Number of neurons in the network.
tau_excitability: Time constant for neuron excitability.
tau_synaptic: Time constant for synaptic plasticity.
dopamine_effect, serotonin_effect, norepinephrine_effect: Effects of neuromodulators.
learning_rate: Learning rate for synaptic weight updates.
time_steps: Number of simulation time steps.
dt: Simulation time step duration.
Code Overview

Simulation Parameters and Initial Conditions
Initial conditions for the neurons, synaptic weights, and neuromodulator levels are set up using biologically plausible values.

Dynamic Baseline Adjustments
The dynamic_baseline function generates a time-varying baseline with controlled fluctuations using an Ornstein-Uhlenbeck process.

Izhikevich Model
The izhikevich_update function updates membrane potentials and recovery variables based on the Izhikevich model.

Neuromodulation
The neuromodulation function calculates the modulation factor based on dopamine, serotonin, and norepinephrine levels.

Synaptic Plasticity
The update_synaptic_weights function implements an enhanced STDP rule with precise spike timing updates.

Network Topology
The create_network_topology function generates a network topology using different models such as small-world, scale-free, or random.

Gradual Update of Initial Conditions
The gradual_update_initial_conditions function smooths adjustments to initial conditions during the simulation.

Error Handling
Comprehensive logging is implemented to capture detailed error information and network state snapshots for effective debugging.

Real-Time Visualization
Real-time visualization of neuron activities is provided using matplotlib.animation.

Performance Optimization
The simulation uses cProfile for profiling and parallel processing for efficient synaptic weight updates.

Example Simulation Code
Here is the most recent simulation code:

import numpy as np
import matplotlib.pyplot as plt
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import Process
from lava.magma.core.resources import CPU
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.io.source import RingBuffer
from lava.proc.io.sink import RingBuffer as RingBufferSink
import traceback
import logging
import networkx as nx
import matplotlib.animation as animation
from multiprocessing import Pool
import cProfile
import pstats

# Setup logging for detailed error information
logging.basicConfig(level=logging.INFO)

# Setting a random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Simulation parameters
num_neurons = 10
tau_excitability = 20
tau_synaptic = 100
dopamine_effect = 0.5
serotonin_effect = 0.3
norepinephrine_effect = 0.2
learning_rate = 0.01
time_steps = 1000
dt = 1

# Initial conditions - using more biologically plausible values
neuron_excitability = np.random.uniform(0.5, 1.5, num_neurons)
synaptic_weights = np.random.rand(num_neurons, num_neurons)
np.fill_diagonal(synaptic_weights, 0)
dopamine_levels = np.random.normal(1.0, 0.1, time_steps)
serotonin_levels = np.random.normal(1.0, 0.1, time_steps)
norepinephrine_levels = np.random.normal(1.0, 0.1, time_steps)
neuron_potentials = np.random.uniform(-65, -55, num_neurons)
recovery_variables = np.random.uniform(-15, -10, num_neurons)

# Dynamic baseline adjustments using Ornstein-Uhlenbeck process
def dynamic_baseline(t, mu=1.0, sigma=0.1, tau=100.0):
    """Generate a time-varying baseline with controlled fluctuations using an Ornstein-Uhlenbeck process."""
    dt = 1
    baseline = np.zeros_like(t)
    baseline[0] = mu
    for i in range(1, len(t)):
        baseline[i] = baseline[i-1] + (mu - baseline[i-1]) * dt / tau + sigma * np.sqrt(dt) * np.random.randn()
    return baseline

t = np.arange(time_steps)
baseline_dopamine = dynamic_baseline(t, mu=1.0, sigma=0.1, tau=100.0)
baseline_serotonin = dynamic_baseline(t, mu=1.0, sigma=0.1, tau=100.0)
baseline_norepinephrine = dynamic_baseline(t, mu=1.0, sigma=0.1, tau=100.0)

def izhikevich_update(V, U, I, a, b, c, d):
    """Update membrane potentials and recovery variables using the Izhikevich model."""
    dV = 0.04 * V**2 + 5 * V + 140 - U + I
    dU = a * (b * V - U)
    V += dV * dt
    U += dU * dt
    spike = V >= 30
    V[spike] = c
    U[spike] += d
    return V, U, spike

def neuromodulation(dopamine_levels, serotonin_levels, norepinephrine_levels, baseline_dopamine, baseline_serotonin, baseline_norepinephrine, dopamine_effect, serotonin_effect, norepinephrine_effect, neuron_indices, t):
    """Calculate modulation factor based on multiple neuromodulators with neuron-specific effects."""
    dopamine_modulation = 1 + dopamine_effect * np.tanh(dopamine_levels[t] - baseline_dopamine[t])
    serotonin_modulation = 1 + serotonin_effect * np.tanh(serotonin_levels[t] - baseline_serotonin[t])
    norepinephrine_modulation = 1 + norepinephrine_effect * np.tanh(norepinephrine_levels[t] - baseline_norepinephrine[t])
    modulation_factors = dopamine_modulation[neuron_indices] * serotonin_modulation[neuron_indices] * norepinephrine_modulation[neuron_indices]
    return modulation_factors

def update_synaptic_weights(weights, spikes_pre, spikes_post, eligibility_traces, learning_rate, tau_eligibility):
    """Enhanced STDP rule with precise spike timing."""
    A_plus = 0.005
    A_minus = 0.005
    tau_plus = 20.0
    tau_minus = 20.0

    pre_indices = np.where(spikes_pre)[0]
    post_indices = np.where(spikes_post)[0]

    for t_pre in pre_indices:
        delta_t = post_indices - t_pre
        ltp_indices = delta_t >= 0
        ltd_indices = delta_t < 0
        eligibility_traces[ltp_indices] += A_plus * np.exp(-delta_t[ltp_indices] / tau_plus)  # LTP
        eligibility_traces[ltd_indices] -= A_minus * np.exp(delta_t[ltd_indices] / tau_minus)  # LTD

    eligibility_traces *= np.exp(-1 / tau_eligibility)
    weights_update = learning_rate * eligibility_traces
    weights += weights_update
    weights = np.clip(weights, 0, 1 - 1e-6)  # Using a small epsilon to avoid abrupt clipping
    return weights, eligibility_traces

def update_neuron_excitability(excitability, modulation_factor, dt, tau_excitability):
    """Update the excitability of neurons."""
    return excitability + dt * (-excitability + modulation_factor) / tau_excitability

def create_network_topology(num_neurons, topology_type="small_world"):
    """Generate a network topology."""
    if topology_type == "small_world":
        G = nx.watts_strogatz_graph(num_neurons, k=4, p=0.1)
    elif topology_type == "scale_free":
        G = nx.scale_free_graph(num_neurons)
    elif topology_type == "random":
        G = nx.erdos_renyi_graph(num_neurons, p=0.1)
    else:
        raise ValueError("Unsupported topology type")
    adj_matrix = nx.adjacency_matrix(G).toarray()
    return adj_matrix

def periodic_network_adjustment(synaptic_weights, target_degree=4, adjustment_rate=0.01):
    """Periodically adjust network connectivity to maintain desired degree distribution."""
    current_degrees = np.sum(synaptic_weights > 0, axis=1)
    for i in range(len(current_degrees)):
        if current_degrees[i] < target_degree:
            candidates = np.where(synaptic_weights[i] == 0)[0]
            if candidates size > 0:
                j = np.random.choice(candidates)
                synaptic_weights[i, j] = np.random.rand() * adjustment_rate
        elif current_degrees[i] > target_degree:
            connected = np.where(synaptic_weights[i] > 0)[0]
            if connected size > 0:
                j = np.random.choice(connected)
                synaptic_weights[i, j] *= 1 - adjustment_rate
    return synaptic_weights

def gradual_update_initial_conditions(t, step_size=0.01):
    global neuron_excitability, synaptic_weights, neuron_potentials, recovery_variables
    if t % 100 == 0:  # Update every 100 time steps
        neuron_excitability += step_size * (np.random.uniform(0.5, 1.5, num_neurons) - neuron_excitability)
        new_synaptic_weights = np.random.rand(num_neurons, num_neurons)
        np.fill_diagonal(new_synaptic_weights, 0)
        synaptic_weights += step_size * (new_synaptic_weights - synaptic_weights)
        neuron_potentials += step_size * (np.random.uniform(-65, -55, num_neurons) - neuron_potentials)
        recovery_variables += step_size * (np.random.uniform(-15, -10, num_neurons) - recovery_variables)

class Neuron(Process):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.input = InPort(shape=shape)
        self.output = OutPort(shape=shape)
        self.voltage = Var(shape=shape, init=np.random.uniform(-65, -55, shape))
        self.recovery = Var(shape=shape, init=np.random.uniform(-15, -10, shape))
        self.excitability = Var(shape=shape, init=np.random.rand(shape))
        self.spikes = Var(shape=shape, init=np.zeros(shape, dtype=bool))

class NeuronModel(PyLoihiProcessModel):
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.input = proc_params["input"]
        self.output = proc_params["output"]
        self.voltage = proc_params["voltage"]
        self.recovery = proc_params["recovery"]
        self.excitability = proc_params["excitability"]
        self.spikes = proc_params["spikes"]

    def run_spk(self):
        try:
            I = self.input.recv()
            modulation_factor = neuromodulation(
                dopamine_levels, serotonin_levels, norepinephrine levels,
                baseline_dopamine, baseline_serotonin, baseline_norepinephrine,
                dopamine_effect, serotonin_effect, norepinephrine_effect,
                np.arange(len(self.voltage)), self.time_step
            )
            self.excitability = update_neuron_excitability(self.excitability, modulation_factor, dt, tau_excitability)
            self.voltage, self.recovery, self.spikes = izhikevich_update(
                self.voltage, self.recovery, I, 0.02, 0.2, -65, 8
            )
            self.output.send(self.spikes)
        except Exception as e:
            logging.error(f"Error in neuron model: {str(e)}")
            logging.error(f"Neuron Potentials: {self.voltage.get()}")
            logging.error(f"Recovery Variables: {self.recovery get()}")
            logging.error(f"Synaptic Weights: {synaptic weights}")
            logging.error(f"Dopamine Levels: {dopamine levels}")
            logging.error(f"Serotonin Levels: {serotonin levels}")
            logging.error(f"Norepinephrine Levels: {norepinephrine levels}")
            logging.error(f"Eligibility Traces: {eligibility traces}")
            traceback.print_exc()

# Instantiate and configure neurons
neurons = Neuron(shape=(num_neurons,))
neurons.in_ports.input.connect(neurons.out_ports.output)

# Create input current generator
input_current = RingBuffer(data=np.random.rand(num_neurons, time_steps).astype(np.float32))

# Create a sink to store output spikes
output_sink = RingBufferSink(shape=(num_neurons,), buffer_len=time_steps)

# Connect the input current to the neuron input
input_current.out_ports.output.connect(neurons.in_ports.input)

# Connect the neuron output to the output sink
neurons.out_ports.output.connect(output_sink.in_ports.input)

# Create a realistic network topology
synaptic_weights = create_network_topology(num_neurons, topology_type="small_world")

# Configure and run the simulation
run_config = Loihi1SimCfg(select_tag="floating_pt", select_sub_proc_model=True)
run_condition = RunSteps(num_steps=time_steps)

try:
    profiler = cProfile.Profile()
    profiler.enable()

    for t in range(time_steps):
        gradual_update_initial_conditions(t)
        synaptic_weights = periodic_network_adjustment(synaptic_weights)
        neurons.run(condition=run_condition, run_cfg=run_config)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
except Exception as e:
    logging.error(f"Error during simulation: {str(e)}")
    logging.error(f"Neuron Model State: {neurons.__dict__}")
    logging.error(f"Neuron Potentials: {neuron_potentials}")
    logging.error(f"Recovery Variables: {recovery_variables}")
    logging.error(f"Synaptic Weights: {synaptic_weights}")
    logging.error(f"Dopamine Levels: {dopamine_levels}")
    logging.error(f"Serotonin Levels: {serotonin_levels}")
    logging.error(f"Norepinephrine Levels: {norepinephrine_levels}")
    logging.error(f"Eligibility Traces: {eligibility_traces}")
    logging.error(f"Network Topology: {create_network_topology(num_neurons, topology_type='small_world')}")
    traceback.print_exc()

# Real-time visualization
fig, ax = plt.subplots()
ax.set_xlim(0, time_steps)
ax.set_ylim(0, num_neurons)
line, = ax.plot([], [], 'r-')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    spike_times = np.where(output_sink.data[:, frame])[0]
    line.set_data(spike_times, np.full_like(spike_times, frame))
    return line,

ani = animation.FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True, interval=50)
plt.show()

# Use a more efficient parallel processing strategy
with Pool(processes=4) as pool:
    results = pool.starmap(update_synaptic_weights, [
        (synaptic_weights, output_sink.data[:, t], output_sink.data[:, t+1], eligibility_traces, learning_rate, tau_eligibility)
        for t in range(time_steps - 1)
    ])
License

This project is licensed under the GNU 3.0 License - see the LICENSE file for details.

Publications that were mandatory for this project's research below:

Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion. Physical Review, 36(5), 823-841. Link

Abbott, L. F., & Regehr, W. G. (2004). Synaptic computation. Nature, 431(7010), 796-803. Link

Markram, H., Gerstner, W., & Sjöström, P. J. (2012). Spike-timing-dependent plasticity: A comprehensive overview. Frontiers in Synaptic Neuroscience, 4, 2. Link

Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nature Neuroscience, 3(9), 919-926. Link

Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442. Link

Special thanks to medical student A.Fogl



