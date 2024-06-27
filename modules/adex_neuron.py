import numpy as np

class AdExNeuron:
    """
    Implements the Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.
    Reference: Brette R, Gerstner W. Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. Journal of Neurophysiology. 2005.
    """
    
    def __init__(self, C, gL, EL, VT, DeltaT, a, tau_w, b, Vr, Vpeak, dt):
        self.C = C
        self.gL = gL
        self.EL = EL
        self.VT = VT
        self.DeltaT = DeltaT
        self.a = a
        self.tau_w = tau_w
        self.b = b
        self.Vr = Vr
        self.Vpeak = Vpeak
        self.dt = dt
        self.V = EL
        self.w = 0

    def validate_parameters(self):
        if not (50 <= self.C <= 500):
            raise ValueError("Capacitance out of range.")
        if not (1 <= self.gL <= 100):
            raise ValueError("Leak conductance out of range.")
        if not (-80 <= self.EL <= -50):
            raise ValueError("Resting potential out of range.")
        if not (-60 <= self.VT <= -40):
            raise ValueError("Threshold potential out of range.")
        if not (0.5 <= self.DeltaT <= 5):
            raise ValueError("Sharpness of exponential approach to threshold out of range.")
        if not (1 <= self.a <= 10):
            raise ValueError("Subthreshold adaptation out of range.")
        if not (10 <= self.tau_w <= 200):
            raise ValueError("Adaptation time constant out of range.")
        if not (0.1 <= self.b <= 50):
            raise ValueError("Spike-triggered adaptation out of range.")
        if not (-70 <= self.Vr <= -40):
            raise ValueError("Reset potential out of range.")
        if not (10 <= self.Vpeak <= 50):
            raise ValueError("Peak potential out of range.")
        if not (0.01 <= self.dt <= 1):
            raise ValueError("Time step out of range.")

    def step(self, I):
        self.validate_parameters()
        dV = (self.gL * (self.EL - self.V) + self.gL * self.DeltaT * np.exp((self.V - self.VT) / self.DeltaT) - self.w + I) / self.C
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w
        self.V += dV * self.dt
        self.w += dw * self.dt
        if self.V >= self.Vpeak:
            self.V = self.Vr
            self.w += self.b
        return self.V, self.w
