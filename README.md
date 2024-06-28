# Neuromorphic Simulation - AC

Introduction
The Neuromorphic Simulation Project aims to simulate neural processes with an emphasis on achieving a concept of artificial consciousness. The simulation incorporates various neural mechanisms such as plasticity, modulation, learning, and emotional states, utilizing state-of-the-art computational techniques and empirical data.


## Directory Structure

```project_root/
├── config/
│   ├── simulation_config.json
│   └── logging_config.json
├── data/
│   ├── neuron_sensitivity.npy
│   ├── initial_conditions.npy
│   ├── cognitive_model_weights.npy
│   ├── dopamine_levels.npy
│   ├── serotonin_levels.npy
│   ├── norepinephrine_levels.npy
│   ├── integration_levels.npy
│   ├── attention_signals.npy
├── docs/
│   └── README.md
├── logs/
│   └── simulation.log
├── modules/
│   ├── baseline.py
│   ├── dehaene_changeux_modulation.py
│   ├── continuous_learning.py
│   ├── emotional_models.py
│   ├── plasticity.py
│   ├── topology.py
│   ├── behavior_monitoring.py
│   ├── self_model.py
│   ├── sensory_motor.py
│   ├── adex_neuron.py
│   ├── ionic_channels.py
├── scripts/
│   ├── generate_data.py
│   ├── run_simulation.py
│   └── visualization.py
└── tests/
    └── test_simulation.py
```

Installation
To run the Neuromorphic Simulation Project, you need Python 3.7+ and the following packages:

numpy
matplotlib
keras
sklearn
scipy
You can install the necessary packages using pip:
```
pip install numpy matplotlib keras sklearn scipy
```
Usage
Generate Data
Before running the simulation, generate the necessary data files by executing the generate_data.py script:
```
python scripts/generate_data.py
```
Run the Simulation
To run the simulation, execute the run_simulation.py script:
```
python scripts/run_simulation.py
```
Visualize Results
To visualize the results of the simulation, use the visualization.py script:
```
python scripts/visualization.py
```


## License

This project is licensed under the GNU 3.0 License - see the LICENSE file for details.

## Publications

- Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion. Physical Review, 36(5), 823-841. [Link](https://link_to_publication)
- Abbott, L. F., & Regehr, W. G. (2004). Synaptic computation. Nature, 431(7010), 796-803. [Link](https://link_to_publication)
- Markram, H., Gerstner, W., & Sjöström, P. J. (2012). Spike-timing-dependent plasticity: A comprehensive overview. Frontiers in Synaptic Neuroscience, 4, 2. [Link](https://link_to_publication)
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nature Neuroscience, 3(9), 919-926. [Link](https://link_to_publication)
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442. [Link](https://link_to_publication)

Special thanks to medical student A. Fogl
```

Modules Description
baseline.py
This module implements dynamic baseline adjustments for neurons based on empirical data.

dehaene_changeux_modulation.py
This module implements the Dehaene-Changeux model for cognitive modulation, including multi-layer integration and hierarchical feedback.

continuous_learning.py
This module implements continuous learning mechanisms, including memory consolidation and adaptive learning rates.

emotional_models.py
This module simulates complex emotional states and includes empathy simulation and mood regulation.

plasticity.py
This module implements synaptic plasticity mechanisms, including STDP and Q-learning updates.

topology.py
This module creates and dynamically adjusts the network topology, including small-world network generation and hierarchical reconfiguration.

behavior_monitoring.py
This module monitors emergent behaviors, detects anomalies, and categorizes and predicts future behaviors using machine learning techniques.

self_model.py
This module simulates self-awareness and decision-making processes, including goal setting and adaptive feedback mechanisms.

sensory_motor.py
This module integrates sensory and motor signals, including adaptive control and dynamic feedback processing.

adex_neuron.py
This module implements the Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.

ionic_channels.py
This module simulates ionic channels in neurons, including state updates and current computation.

Configuration Files
simulation_config.json
This file contains configuration parameters for the simulation, including the number of neurons, time steps, learning rates, and topology settings.

logging_config.json
This file contains the logging configuration for the simulation, defining log levels and handlers.

Data Files
The data directory contains the following data files needed for the simulation:

neuron_sensitivity.npy: Neuron sensitivity data
initial_conditions.npy: Initial conditions for the simulation
cognitive_model_weights.npy: Weights for the cognitive model
dopamine_levels.npy: Dopamine levels over time
serotonin_levels.npy: Serotonin levels over time
norepinephrine_levels.npy: Norepinephrine levels over time
integration_levels.npy: Integration levels for the modulation model
attention_signals.npy: Attention signals for the modulation model
Scripts
generate_data.py
This script generates the necessary data files for the simulation based on empirical distributions.

run_simulation.py
This script sets up and runs the simulation, initializing all modules and executing the simulation loop.

visualization.py
This script visualizes the results of the simulation, including neural activity, synaptic weights, and emotional states.
