import numpy as np
import os
from scipy.stats import norm

# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Generate neuron sensitivity data based on empirical ranges
neuron_sensitivity = norm.rvs(loc=1.0, scale=0.1, size=(100, 3))
np.save('data/neuron_sensitivity.npy', neuron_sensitivity)

# Generate initial conditions based on empirical data
initial_conditions = {
    'neuron_excitability': norm.rvs(loc=1.0, scale=0.3, size=100),
    'synaptic_weights': np.random.rand(100, 100),
    'neuron_potentials': norm.rvs(loc=-60, scale=5, size=100),
    'recovery_variables': norm.rvs(loc=-12.5, scale=2.5, size=100)
}
np.save('data/initial_conditions.npy', initial_conditions)

# Generate cognitive model weights (example)
cognitive_model_weights = np.random.rand(100, 100)
np.save('data/cognitive_model_weights.npy', cognitive_model_weights)

# Generate neuromodulator levels
dopamine_levels = norm.rvs(loc=1.0, scale=0.25, size=1000)
serotonin_levels = norm.rvs(loc=1.0, scale=0.25, size=1000)
norepinephrine_levels = norm.rvs(loc=1.0, scale=0.25, size=1000)
integration_levels = norm.rvs(loc=1.0, scale=0.25, size=1000)
attention_signals = norm.rvs(loc=1.0, scale=0.25, size=1000)

np.save('data/dopamine_levels.npy', dopamine_levels)
np.save('data/serotonin_levels.npy', serotonin_levels)
np.save('data/norepinephrine_levels.npy', norepinephrine_levels)
np.save('data/integration_levels.npy', integration_levels)
np.save('data/attention_signals.npy', attention_signals)

print("Data generated and saved.")
