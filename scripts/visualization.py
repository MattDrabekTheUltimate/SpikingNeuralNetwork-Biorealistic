import numpy as np
import matplotlib.pyplot as plt

def visualize_neural_activity(activity_data, title="Neural Activity"):
    plt.figure(figsize=(10, 6))
    plt.plot(activity_data)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Activity")
    plt.show()

def visualize_synaptic_weights(weights, title="Synaptic Weights"):
    plt.figure(figsize=(10, 6))
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_emotional_state(emotional_state, title="Emotional State"):
    plt.figure(figsize=(10, 6))
    labels = list(emotional_state.keys())
    values = list(emotional_state.values())
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Emotional State Variables")
    plt.ylabel("Values")
    plt.show()

if __name__ == "__main__":
    # Example visualization calls
    neural_activity = np.random.rand(1000)
    synaptic_weights = np.random.rand(100, 100)
    emotional_state = {'happiness': 0.8, 'stress': 0.2, 'motivation': 0.5}

    visualize_neural_activity(neural_activity)
    visualize_synaptic_weights(synaptic_weights)
    visualize_emotional_state(emotional_state)
