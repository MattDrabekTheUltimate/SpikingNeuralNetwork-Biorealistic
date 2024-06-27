import numpy as np

def generate_neuron_sensitivity():
    neuron_sensitivity = np.random.uniform(0.8, 1.2, (100, 3))
    np.save('data/neuron_sensitivity.npy', neuron_sensitivity)

def generate_initial_conditions():
    initial_conditions = {
        'neuron_excitability': np.random.uniform(0.5, 1.5, 100),
        'synaptic_weights': np.random.rand(100, 100),
        'neuron_potentials': np.random.uniform(-65, -55, 100),
        'recovery_variables': np.random.uniform(-15, -10, 100)
    }
    np.save('data/initial_conditions.npy', initial_conditions)

def generate_cognitive_model_weights():
    cognitive_model_weights = np.random.rand(100, 100)
    np.save('data/cognitive_model_weights.npy', cognitive_model_weights)

def generate_neuromodulator_levels():
    dopamine_levels = np.random.uniform(0.5, 1.5, 1000)
    serotonin_levels = np.random.uniform(0.5, 1.5, 1000)
    norepinephrine_levels = np.random.uniform(0.5, 1.5, 1000)
    integration_levels = np.random.uniform(0.5, 1.5, 100)
    attention_signals = np.random.uniform(0.5, 1.5, 100)

    np.save('data/dopamine_levels.npy', dopamine_levels)
    np.save('data/serotonin_levels.npy', serotonin_levels)
    np.save('data/norepinephrine_levels.npy', norepinephrine_levels)
    np.save('data/integration_levels.npy', integration_levels)
    np.save('data/attention_signals.npy', attention_signals)

def main():
    generate_neuron_sensitivity()
    generate_initial_conditions()
    generate_cognitive_model_weights()
    generate_neuromodulator_levels()

if __name__ == '__main__':
    main()
