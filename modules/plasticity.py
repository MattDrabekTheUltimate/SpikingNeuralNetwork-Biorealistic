import logging
import numpy as np

logger = logging.getLogger(__name__)

def update_synaptic_weights(weights, spikes_pre, spikes_post, eligibility_traces, learning_rate, tau_eligibility):
    A_plus = 0.005
    A_minus = 0.005
    tau_plus = 20.0
    tau_minus = 20.0

    pre_indices = np.where(spikes_pre)[0]
    post_indices = np.where(spikes_post)[0]
    delta_t = post_indices[:, np.newaxis] - pre_indices
    ltp = (delta_t >= 0).astype(float) * A_plus * np.exp(-delta_t / tau_plus)
    ltd = (delta_t < 0).astype(float) * A_minus * np.exp(delta_t / tau_minus)
    eligibility_traces += np.sum(ltp, axis=0) - np.sum(ltd, axis=0)
    eligibility_traces *= np.exp(-1 / tau_eligibility)
    weights += learning_rate * eligibility_traces
    weights = np.clip(weights, 0, 1 - 1e-6)
    
    logger.debug(f"Updated weights: {weights}")
    logger.debug(f"Eligibility traces: {eligibility_traces}")

    return weights, eligibility_traces
