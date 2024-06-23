import numpy as np

def dehaene_changeux_modulation(neuron_activity, integration_levels, attention_signals, cognitive_model_weights):
    # Normalize inputs
    neuron_activity_normalized = neuron_activity / np.max(neuron_activity)
    integration_levels_normalized = integration_levels / np.max(integration_levels)
    attention_signals_normalized = attention_signals / np.max(attention_signals)
    
    # Apply cognitive model weights
    integrated_activity = np.dot(cognitive_model_weights, neuron_activity_normalized)
    integrated_activity = integrated_activity * integration_levels_normalized * attention_signals_normalized
    
    return integrated_activity
