import numpy as np

class IonicChannel:
    def __init__(self, g_max, E_rev, alpha, beta):
        self.g_max = g_max
        self.E_rev = E_rev
        self.alpha = alpha
        self.beta = beta
        self.state = 0

    def update_state(self, V):
        self.state += self.alpha * (1 - self.state) - self.beta * self.state
        return self.state

    def compute_current(self, V):
        self.update_state(V)
        I = self.g_max * self.state * (V - self.E_rev)
        return I

    def validate_parameters(self):
        if not (1 <= self.g_max <= 200):
            raise ValueError("Maximum conductance out of range.")
        if not (-100 <= self.E_rev <= 100):
            raise ValueError("Reversal potential out of range.")
        if not (0 <= self.alpha <= 1):
            raise ValueError("Alpha parameter out of range.")
        if not (0 <= self.beta <= 1):
            raise ValueError("Beta parameter out of range.")
