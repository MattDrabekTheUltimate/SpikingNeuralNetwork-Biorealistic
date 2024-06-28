import numpy as np
import json
import os

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def generate_neuron_sensitivity(num_neurons):
    return np.random.uniform(0.8, 1.2, num_neurons)

def generate_initial_conditions(num_neurons):
    return {
        'neuron_excitability': np.random.uniform(0.5, 1.5, num_neurons),
        'synaptic_weights': np.random.rand(num_neurons, num_neurons),
        'neuron_potentials': np.random.uniform(-65, -55, num_neurons),
        'recovery_variables': np.random.uniform(-15, -10, num_neurons)
    }

def generate_cognitive_model_weights(num_neurons):
    return np.random.rand(num_neurons, num_neurons)

def generate_neuromodulator_levels(time_steps):
    return {
        'dopamine_levels': np.random.uniform(0.5, 1.5, time_steps),
        'serotonin_levels': np.random.uniform(0.5, 1.5, time_steps),
        'norepinephrine_levels': np.random.uniform(0.5, 1.5, time_steps),
        'integration_levels': np.random.uniform(0.5, 1.5, time_steps),
        'attention_signals': np.random.uniform(0.5, 1.5, time_steps)
    }

with open('config/simulation_config.json', 'r') as f:
    config = json.load(f)

num_neurons = config['num_neurons']
time_steps = config['time_steps']

neuron_sensitivity = generate_neuron_sensitivity(num_neurons)
initial_conditions = generate_initial_conditions(num_neurons)
cognitive_model_weights = generate_cognitive_model_weights(num_neurons)
neuromodulator_levels = generate_neuromodulator_levels(time_steps)

np.save(os.path.join(data_dir, 'neuron_sensitivity.npy'), neuron_sensitivity)
np.save(os.path.join(data_dir, 'initial_conditions.npy'), initial_conditions)
np.save(os.path.join(data_dir, 'cognitive_model_weights.npy'), cognitive_model_weights)
np.save(os.path.join(data_dir, 'dopamine_levels.npy'), neuromodulator_levels['dopamine_levels'])
np.save(os.path.join(data_dir, 'serotonin_levels.npy'), neuromodulator_levels['serotonin_levels'])
np.save(os.path.join(data_dir, 'norepinephrine_levels.npy'), neuromodulator_levels['norepinephrine_levels'])
np.save(os.path.join(data_dir, 'integration_levels.npy'), neuromodulator_levels['integration_levels'])
np.save(os.path.join(data_dir, 'attention_signals.npy'), neuromodulator_levels['attention_signals'])
