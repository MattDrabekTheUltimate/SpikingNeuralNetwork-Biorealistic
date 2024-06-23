import numpy as np

def neuromodulation(dopamine_levels, serotonin_levels, norepinephrine_levels, baseline_dopamine, baseline_serotonin, baseline_norepinephrine, dopamine_effect, serotonin_effect, norepinephrine_effect, t, num_neurons, neuron_sensitivity):
    modulation_factors = np.ones(num_neurons)
    for i in range(num_neurons):
        dopamine_modulation = 1 + dopamine_effect * neuron_sensitivity[i, 0] * np.tanh(dopamine_levels[t] - baseline_dopamine[t])
        serotonin_modulation = 1 + serotonin_effect * neuron_sensitivity[i, 1] * np.tanh(serotonin_levels[t] - baseline_serotonin[t])
        norepinephrine_modulation = 1 + norepinephrine_effect * neuron_sensitivity[i, 2] * np.tanh(norepinephrine_levels[t] - baseline_norepinephrine[t])
        modulation_factors[i] = dopamine_modulation * serotonin_modulation * norepinephrine_modulation
        if not (0.8 <= modulation_factors[i] <= 1.2):
            raise ValueError(f"Modulation factor for neuron {i} is out of realistic range.")
    return modulation_factors
