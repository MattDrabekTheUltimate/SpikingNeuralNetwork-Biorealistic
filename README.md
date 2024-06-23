# Neuromorphic Simulation

This project simulates a neuromorphic system with dynamic baseline, neuromodulation, plasticity, and network topology.

## Directory Structure

```project_root/
├── config/
│   ├── simulation_config.json
│   └── logging_config.json
├── data/
│   ├── neuron_sensitivity.npy
│   ├── initial_conditions.npy
│   └── cognitive_model_weights.npy
├── docs/
│   └── README.md
├── logs/
│   └── simulation.log
├── modules/
│   ├── baseline.py
│   ├── neuromodulation.py
│   ├── plasticity.py
│   ├── topology.py
│   └── cognitive_integration.py
├── scripts/
│   ├── generate_data.py
│   ├── run_simulation.py
│   └── visualization.py
└── tests/
    └── test_simulation.py
```



## Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- Vispy
- Lava (Loihi simulator)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository

Install the required packages:

pip install -r requirements.txt


Running the Simulation

Generate data:
python scripts/generate_data.py


Run the simulation:
python scripts/run_simulation.py

Visualize the results:
python scripts/visualization.py

Running Tests:
python -m unittest discover tests

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
