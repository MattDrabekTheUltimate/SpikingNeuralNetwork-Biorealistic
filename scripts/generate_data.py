import numpy as np
import os

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Generate neuron sensitivity data based on empirical ranges
neuron_sensitivity = np.random.uniform(0.8, 1.2, (100, 3))
np.save('data/neuron_sensitivity.npy', neuron_sensitivity)

# Generate initial conditions based on empirical data
initial_conditions = {
    'neuron_excitability': np.random.uniform(0.5, 1.5, 100),
    'synaptic_weights': np.random.rand(100, 100),
    'neuron_potentials': np.random.uniform(-65, -55, 100),
    'recovery_variables': np.random.uniform(-15, -10, 100)
}
np.save('data/initial_conditions.npy', initial_conditions)

# Generate cognitive model weights (example, replace with actual empirical data if available)
cognitive_model_weights = np.random.rand(100, 100)
np.save('data/cognitive_model_weights.npy', cognitive_model_weights)

print("Neuron sensitivity, initial conditions, and cognitive model weights generated and saved.")
