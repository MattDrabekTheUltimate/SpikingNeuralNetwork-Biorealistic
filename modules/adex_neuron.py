import numpy as np

class AdExNeuron:
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
        self.V = EL  # Membrane potential
        self.w = 0  # Adaptation variable

    def step(self, I):
        dV = (self.gL * (self.EL - self.V) + self.gL * self.DeltaT * np.exp((self.V - self.VT) / self.DeltaT) - self.w + I) / self.C
        dw = (self.a * (self.V - self.EL) - self.w) / self.tau_w
        self.V += dV * self.dt
        self.w += dw * self.dt
        if self.V >= self.Vpeak:
            self.V = self.Vr
            self.w += self.b
        return self.V, self.w
