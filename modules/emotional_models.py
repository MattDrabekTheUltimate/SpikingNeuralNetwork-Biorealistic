import numpy as np

class EmotionalModel:
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

    def empathy_simulation(self, external_emotional_states):
        for external_state in external_emotional_states:
            self.state['happiness'] += external_state['happiness'] * 0.1
            self.state['stress'] += external_state['stress'] * 0.1
            self.state['motivation'] += external_state['motivation'] * 0.1

    def mood_regulation(self, regulation_factor):
        self.state['happiness'] *= (1 + regulation_factor)
        self.state['stress'] *= (1 - regulation_factor)
