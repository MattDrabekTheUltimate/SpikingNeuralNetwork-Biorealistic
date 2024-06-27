import numpy as np

class EmotionalModel:
    """
    Implements an emotional model for simulating complex emotional states.
    Reference: Dayan P, Huys QJ. Serotonin in affective control. Annual Review of Neuroscience. 2009.
    """
    
    def __init__(self, initial_state):
        self.state = initial_state
        self.history = []

    def update_state_dynamic(self, external_factors, internal_feedback):
        self.state['happiness'] += external_factors['positive_events'] - internal_feedback['stress']
        self.state['stress'] += external_factors['negative_events'] - internal_feedback['calmness']
        self.state['motivation'] += external_factors['goals_achieved'] - internal_feedback['frustration']

    def simulate_complex_emotional_states(self, dopamine_levels, serotonin_levels, norepinephrine_levels, t):
        self.state['happiness'] = np.mean(dopamine_levels[t])
        self.state['calmness'] = np.mean(serotonin_levels[t])
        self.state['alertness'] = np.mean(norepinephrine_levels[t])
        self.state['stress'] = 1.0 / (np.mean(serotonin_levels[t]) + np.mean(dopamine_levels[t]))
        self.state['motivation'] = np.mean(dopamine_levels[t]) * 0.5 + np.mean(norepinephrine_levels[t]) * 0.5
        self.state['frustration'] = 1.0 / (np.mean(dopamine_levels[t]) * np.mean(norepinephrine_levels[t]))
        self.state['satisfaction'] = np.mean(dopamine_levels[t]) - np.std(dopamine_levels[t])
        self.history.append(self.state.copy())

    def get_emotional_state(self):
        return self.state

    def get_emotional_history(self):
        return self.history
