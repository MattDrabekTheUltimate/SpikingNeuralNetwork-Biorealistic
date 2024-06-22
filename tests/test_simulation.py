import unittest
import numpy as np
from modules.baseline import dynamic_baseline
from modules.neuromodulation import neuromodulation
from modules.plasticity import update_synaptic_weights
from modules.topology import create_network_topology, dynamic_topology_switching

class TestSimulation(unittest.TestCase):

    def test_dynamic_baseline(self):
        t = np.arange(100)
        baseline = dynamic_baseline(t)
        self.assertEqual(len(baseline), 100)

    def test_neuromodulation(self):
        t = 0
        num_neurons = 100
        neuron_sensitivity = np.random.uniform(0.8, 1.2, (100, 3))
        dopamine_levels = np.random.rand(100)
        serotonin_levels = np.random.rand(100)
        norepinephrine_levels = np.random.rand(100)
        baseline_dopamine = np.random.rand(100)
        baseline_serotonin = np.random.rand(100)
        baseline_norepinephrine = np.random.rand(100)
        modulation_factors = neuromodulation(dopamine_levels, serotonin_levels, norepinephrine_levels, baseline_dopamine, baseline_serotonin, baseline_norepinephrine, 0.1, 0.1, 0.1, t, num_neurons, neuron_sensitivity)
        self.assertEqual(len(modulation_factors), num_neurons)

    def test_update_synaptic_weights(self):
        weights = np.random.rand(100, 100)
        spikes_pre = np.random.rand(100) > 0.5
        spikes_post = np.random.rand(100) > 0.5
        eligibility_traces = np.zeros(100)
        learning_rate = 0.01
        tau_eligibility = 20.0
        updated_weights, updated_eligibility_traces = update_synaptic_weights(weights, spikes_pre, spikes_post, eligibility_traces, learning_rate, tau_eligibility)
        self.assertEqual(updated_weights.shape, (100, 100))

    def test_create_network_topology(self):
        synaptic_weights = create_network_topology(100)
        self.assertEqual(synaptic_weights.shape, (100, 100))

    def test_dynamic_topology_switching(self):
        synaptic_weights = create_network_topology(100)
        spikes = np.random.rand(100) > 0.5
        updated_weights = dynamic_topology_switching(synaptic_weights, spikes)
        self.assertEqual(updated_weights.shape, (100, 100))

if __name__ == '__main__':
    unittest.main()
