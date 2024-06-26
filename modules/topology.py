import numpy as np

def create_network_topology(num_neurons, topology_type="small_world", p_rewire=0.1, k=4):
    """
    Creates a neural network topology.
    Reference: Sporns O, Chialvo DR, Kaiser M, Hilgetag CC. Organization, development and function of complex brain networks. Trends in Cognitive Sciences. 2004.
    """
    if topology_type == "small_world":
        synaptic_weights = np.zeros((num_neurons, num_neurons))
        for i in range(num_neurons):
            for j in range(1, k // 2 + 1):
                synaptic_weights[i, (i + j) % num_neurons] = 1
                synaptic_weights[i, (i - j) % num_neurons] = 1
        for i in range(num_neurons):
            for j in range(i + 1, num_neurons):
                if synaptic_weights[i, j] == 1 and np.random.rand() < p_rewire:
                    new_connection = np.random.randint(0, num_neurons)
                    while new_connection == i or synaptic_weights[i, new_connection] == 1:
                        new_connection = np.random.randint(0, num_neurons)
                    synaptic_weights[i, j] = 0
                    synaptic_weights[i, new_connection] = 1
    return synaptic_weights

def dynamic_topology_switching(synaptic_weights, spikes, target_degree=4, adjustment_rate=0.01):
    """
    Implements dynamic network reconfiguration.
    Reference: Sporns O, Chialvo DR, Kaiser M, Hilgetag CC. Organization, development and function of complex brain networks. Trends in Cognitive Sciences. 2004.
    """
    current_degrees = np.sum(synaptic_weights > 0, axis=1)
    for i in range(len(current_degrees)):
        if current_degrees[i] < target_degree:
            candidates = np.where(synaptic_weights[i] == 0)[0]
            if candidates.size > 0:
                j = np.random.choice(candidates)
                synaptic_weights[i, j] = np.random.rand() * adjustment_rate
        elif current_degrees[i] > target_degree:
            connected = np.where(synaptic_weights[i] > 0)[0]
            if connected.size > 0:
                j = np.random.choice(connected)
                synaptic_weights[i, j] *= 1 - adjustment_rate
    return synaptic_weights
