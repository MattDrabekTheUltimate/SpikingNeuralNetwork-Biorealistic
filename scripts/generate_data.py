import numpy as np

# Generate neuron sensitivity data
neuron_sensitivity = np.random.uniform(0.8, 1.2, (100, 3))
np.save('data/neuron_sensitivity.npy', neuron_sensitivity)

# Generate initial conditions
initial_conditions = {
    'neuron_excitability': np.random.uniform(0.5, 1.5, 100),
    'synaptic_weights': np.random.rand(100, 100),
    'neuron_potentials': np.random.uniform(-65, -55, 100),
    'recovery_variables': np.random.uniform(-15, -10, 100)
}
np.save('data/initial_conditions.npy', initial_conditions)
