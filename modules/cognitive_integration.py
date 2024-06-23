import numpy as np

# Reference: "A hierarchical neural network model of the primate visual system" - Journal of Neuroscience (2011)
def dehaene_changeux_modulation(neuron_activity, integration_levels, attention_signals, cognitive_model_weights):
    # Normalize inputs
    neuron_activity_normalized = neuron_activity / np.max(neuron_activity)
    integration_levels_normalized = integration_levels / np.max(integration_levels)
    attention_signals_normalized = attention_signals / np.max(attention_signals)
    
    # Apply cognitive model weights
    integrated_activity = np.dot(cognitive_model_weights, neuron_activity_normalized)
    integrated_activity = integrated_activity * integration_levels_normalized * attention_signals_normalized
    
    # Implement hierarchical feedback
    feedback_strength = 0.1  # Example value, adjust based on empirical data
    hierarchical_feedback = feedback_strength * np.tanh(integrated_activity - neuron_activity_normalized)
    integrated_activity += hierarchical_feedback
    
    return integrated_activity
