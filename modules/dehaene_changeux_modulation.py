import numpy as np
from modules.continuous_learning import ContinuousLearning

class DehaeneChangeuxModulation:
    """
    Implements the Dehaene-Changeux model for cognitive modulation.
    Reference: Rolls ET. A hierarchical neural network model of the primate visual system. Journal of Neuroscience. 2011.
    """

    def __init__(self, neuron_count, layer_count, noise_level=0.05):
        self.neuron_count = neuron_count
        self.layer_count = layer_count
        self.noise_level = noise_level
        self.cognitive_model_weights = np.random.rand(neuron_count, neuron_count)
        self.integration_levels = np.random.rand(neuron_count)
        self.attention_signals = np.random.rand(neuron_count)
        self.neural_noise = np.random.randn(neuron_count) * noise_level
        self.continuous_learning = ContinuousLearning()

    def set_parameters(self, cognitive_model_weights=None, integration_levels=None, attention_signals=None):
        if cognitive_model_weights is not None:
            self.cognitive_model_weights = cognitive_model_weights
        if integration_levels is not None:
            self.integration_levels = integration_levels
        if attention_signals is not None:
            self.attention_signals = attention_signals

    def normalize(self, array):
        return (array - np.mean(array)) / np.std(array)
    
    def multi_layer_integration(self, activity):
        layer_activities = []
        for _ in range(self.layer_count):
            activity = np.dot(self.cognitive_model_weights, activity)
            activity = self.normalize(activity) * self.integration_levels
            activity = np.tanh(activity)
            activity = activity * self.attention_signals
            layer_activities.append(activity)
        return layer_activities[-1]
    
    def empirical_feedback(self, integrated_activity, neuron_activity):
        feedback_strength_base = 0.1
        feedback_strength_dynamic = feedback_strength_base * np.std(neuron_activity)
        return feedback_strength_dynamic * np.tanh(integrated_activity - neuron_activity)
    
    def add_neural_noise(self, integrated_activity):
        return integrated_activity + self.neural_noise
    
    def update_weights(self, integrated_activity):
        self.cognitive_model_weights += self.continuous_learning.learn_from_experience(integrated_activity)
        self.cognitive_model_weights = np.clip(self.cognitive_model_weights, -1, 1)

    def modulate_activity(self, neuron_activity):
        normalized_activity = self.normalize(neuron_activity)
        integrated_activity = self.multi_layer_integration(normalized_activity)
        hierarchical_feedback = self.empirical_feedback(integrated_activity, normalized_activity)
        integrated_activity += hierarchical_feedback
        integrated_activity = self.add_neural_noise(integrated_activity)
        self.update_weights(integrated_activity)
        return integrated_activity

