import numpy as np

def dynamic_baseline(t, mu=1.0, sigma=0.005, tau=100.0, kp=0.1, ki=0.01, kd=0.01, network_state=None):
    error = sigma
    integral = 0
    derivative = 0
    last_error = 0

    baseline = mu + sigma * np.random.randn(len(t))
    for i in range(1, len(t)):
        if network_state is not None:
            # Adjust sigma based on the network's firing rate variability
            firing_rate_variability = np.var(network_state)
            sigma = sigma + kp * firing_rate_variability
        error = sigma - baseline[i-1]
        integral += error
        derivative = error - last_error
        adjustment = kp * error + ki * integral + kd * derivative
        baseline[i] += adjustment
        last_error = error

    return baseline
