import numpy as np

def sensory_motor_integration(sensory_input, motor_output, integration_params):
    """
    Implements sensory and motor integration in neurons.
    Reference: Evarts EV. Relation of pyramidal tract activity to force exerted during voluntary movement. Journal of Neurophysiology. 1968.
    """
    integrated_response = np.zeros_like(sensory_input)
    for i in range(len(sensory_input)):
        integrated_response[i] = integration_params['gain'] * sensory_input[i] + integration_params['bias'] * motor_output[i]
    return integrated_response

def nonlinear_integration(self, sensory_input, motor_output):
    """
    Implements non-linear dynamics in sensory-motor integration.
    Reference: Evarts EV. Relation of pyramidal tract activity to force exerted during voluntary movement. Journal of Neurophysiology. 1968.
    """
    integrated_response = np.tanh(self.integration_params['gain'] * sensory_input + self.integration_params['bias'] * motor_output)
    return integrated_response
