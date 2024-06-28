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


Modules Description

baseline.py
- This module implements dynamic baseline adjustments for neurons based on empirical data.

dehaene_changeux_modulation.py
- This module implements the Dehaene-Changeux model for cognitive modulation, including multi-layer integration and hierarchical feedback.

continuous_learning.py
- This module implements continuous learning mechanisms, including memory consolidation and adaptive learning rates.

emotional_models.py
- This module simulates complex emotional states and includes empathy simulation and mood regulation.

plasticity.py
- This module implements synaptic plasticity mechanisms, including STDP and Q-learning updates.

topology.py
- This module creates and dynamically adjusts the network topology, including small-world network generation and hierarchical reconfiguration.

behavior_monitoring.py
- This module monitors emergent behaviors, detects anomalies, and categorizes and predicts future behaviors using machine learning techniques.

self_model.py
- This module simulates self-awareness and decision-making processes, including goal setting and adaptive feedback mechanisms.

sensory_motor.py
- This module integrates sensory and motor signals, including adaptive control and dynamic feedback processing.

adex_neuron.py
- This module implements the Adaptive Exponential Integrate-and-Fire (AdEx) neuron model.

ionic_channels.py
- This module simulates ionic channels in neurons, including state updates and current computation.


Configuration Files

simulation_config.json
- This file contains configuration parameters for the simulation, including the number of neurons, time steps, learning rates, and topology settings.

logging_config.json
- This file contains the logging configuration for the simulation, defining log levels and handlers.

Data Files
The data directory contains the following data files needed for the simulation:

- neuron_sensitivity.npy: Neuron sensitivity data
- initial_conditions.npy: Initial conditions for the simulation
- cognitive_model_weights.npy: Weights for the cognitive model
- dopamine_levels.npy: Dopamine levels over time
- serotonin_levels.npy: Serotonin levels over time
- norepinephrine_levels.npy: Norepinephrine levels over time
- integration_levels.npy: Integration levels for the modulation model
- attention_signals.npy: Attention signals for the modulation model

Scripts
generate_data.py
This script generates the necessary data files for the simulation based on empirical distributions.

run_simulation.py
This script sets up and runs the simulation, initializing all modules and executing the simulation loop.

visualization.py
This script visualizes the results of the simulation, including neural activity, synaptic weights, and emotional states.


## License

This project is licensed under the GNU 3.0 License - see the LICENSE file for details.

### Publications

1. **Uhlenbeck, G. E., & Ornstein, L. S. (1930).** On the theory of the Brownian motion. Physical Review, 36(5), 823-841. [Link](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823)

2. **Abbott, L. F., & Regehr, W. G. (2004).** Synaptic computation. Nature, 431(7010), 796-803. [Link](https://www.nature.com/articles/nature03010)

3. **Markram, H., Gerstner, W., & Sjöström, P. J. (2012).** Spike-timing-dependent plasticity: A comprehensive overview. Frontiers in Synaptic Neuroscience, 4, 2. [Link](https://www.frontiersin.org/articles/10.3389/fnsyn.2012.00002/full)

4. **Song, S., Miller, K. D., & Abbott, L. F. (2000).** Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nature Neuroscience, 3(9), 919-926. [Link](https://www.nature.com/articles/nn0900_919)

5. **Watts, D. J., & Strogatz, S. H. (1998).** Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442. [Link](https://www.nature.com/articles/30918)

6. **Turrigiano, G. G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998).** Activity-dependent scaling of quantal amplitude in neocortical neurons. Nature, 391(6670), 892-896. [Link](https://www.nature.com/articles/36103)

7. **Schultz, W. (1998).** Predictive reward signal of dopamine neurons. Journal of Neurophysiology, 80(1), 1-27. [Link](https://journals.physiology.org/doi/full/10.1152/jn.1998.80.1.1)

8. **Gross, G. W., Rhoades, B. K., Azzazy, H. M., & Wu, M. C. (1995).** The use of neuronal networks on multielectrode arrays as biosensors. Biosensors and Bioelectronics, 10(6-7), 553-567. [Link](https://www.sciencedirect.com/science/article/abs/pii/095656639500049C)

9. **Sejnowski, T. J., Koch, C., & Churchland, P. S. (2014).** Computational neuroscience: Realistic modeling for experimentalists. Nature Neuroscience, 3(11), 1164-1171. [Link](https://www.nature.com/articles/nn1100_1164)

10. **Izhikevich, E. M. (2007).** Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting. MIT Press. [Link](https://mitpress.mit.edu/books/dynamical-systems-neuroscience)

11. **Rolls, E. T. (2011).** A hierarchical neural network model of the primate visual system. Journal of Neuroscience, 31(19), 7192-7210. [Link](https://www.jneurosci.org/content/31/19/7192)

12. **Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).** Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533. [Link](https://www.nature.com/articles/nature14236)

13. **Dayan, P., & Huys, Q. J. (2009).** Serotonin in affective control. Annual Review of Neuroscience, 32, 95-126. [Link](https://www.annualreviews.org/doi/abs/10.1146/annurev.neuro.051508.135607)

14. **Damasio, A. R. (1999).** The Feeling of What Happens: Body and Emotion in the Making of Consciousness. Harcourt Brace.

15. **Bandura, A. (1997).** Self-efficacy: The exercise of control. W.H. Freeman.

16. **Carver, C. S., & Scheier, M. F. (1982).** Control theory: A useful conceptual framework for personality-social, clinical, and health psychology. Psychological Bulletin, 92(1), 111-135. [Link](https://psycnet.apa.org/doi/10.1037/0033-2909.92.1.111)

17. **Evarts, E. V. (1968).** Relation of pyramidal tract activity to force exerted during voluntary movement. Journal of Neurophysiology, 31(1), 14-27. [Link](https://journals.physiology.org/doi/abs/10.1152/jn.1968.31.1.14)

18. **Shadmehr, R., Smith, M. A., & Krakauer, J. W. (2010).** Error correction, sensory prediction, and adaptation in motor control. Annual Review of Neuroscience, 33, 89-108. [Link](https://www.annualreviews.org/doi/10.1146/annurev-neuro-060909-153135)

19. **Faisal, A. A., Selen, L. P., & Wolpert, D. M. (2008).** Noise in the nervous system. Nature Reviews Neuroscience, 9(4), 292-303. [Link](https://www.nature.com/articles/nrn2258)

20. **Magee, J. C., & Johnston, D. (2005).** Plasticity of dendritic function. Current Opinion in Neurobiology, 15(3), 334-342. [Link](https://www.sciencedirect.com/science/article/abs/pii/S095943880500077X)

21. **Kohonen, T. (2001).** Self-Organizing Maps. Springer. [Link](https://link.springer.com/book/10.1007/978-3-642-56927-2)

22. **Robbins, T. W., & Everitt, B. J. (1995).** Central norepinephrine neurons and behavior. Psychopharmacology, 118(4), 393-408. [Link](https://link.springer.com/article/10.1007/BF02246309)

23. **Frackowiak, R. S., Ashburner, J., Penny, W. D., Zeki, S., & Friston, K. (2004).** Human Brain Function. Elsevier Academic Press. [Link](https://www.elsevier.com/books/human-brain-function/frackowiak/978-0-12-264841-0)

24. **Sherrington, C. S. (1906).** The Integrative Action of the Nervous System. Yale University Press.

25. **Posner, M. I., & Dehaene, S. (1994).** Attentional networks. Trends in Neurosciences, 17(2), 75-79. [Link](https://www.sciencedirect.com/science/article/abs/pii/0166223694901599)

26. **Hille, B. (2001).** Ion Channels of Excitable Membranes. Sinauer Associates. [Link](https://www.sinauer.com/ion-channels-of-excitable-membranes-9780878933211.html)

27. **Caporale, N., & Dan, Y. (2008).** Spike-timing-dependent plasticity: A Hebbian learning rule. Annual Review of Neuroscience, 31, 25-46. [Link](https://www.annualreviews.org/doi/10.1146/annurev.neuro.31.060407.125639)

28. **Gurney, K., Prescott, T. J., & Redgrave, P. (2001).** The basal ganglia: A vertebrate solution to the selection problem? Neuroscience, 25(3), 421-450. [Link](https://www.sciencedirect.com/science/article/abs/pii/S0896627301004101)

29. **Sporns, O., Chialvo, D. R., Kaiser, M., & Hilgetag, C. C. (2004).** Organization, development and function of complex brain networks. Trends in Cognitive Sciences, 8(9), 418-425. [Link](https://www.sciencedirect.com/science/article/abs/pii/S1364661304001858)

30. **Sutton, R. S., & Barto, A. G. (2018).** Reinforcement Learning: An Introduction. MIT Press. [Link](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)

31. **Shumway, R. H., & Stoffer, D. S. (201

0).** Time Series Analysis and Its Applications: With R Examples. Springer. [Link](https://www.springer.com/gp/book/9781441978646)

32. **Metzinger, T. (2009).** The Ego Tunnel: The Science of the Mind and the Myth of the Self. Basic Books. [Link](https://www.basicbooks.com/titles/thomas-metzinger/the-ego-tunnel/9780465045679/)

33. **Desmurget, M., & Grafton, S. (2000).** Forward modeling allows feedback control for fast reaching movements. Trends in Cognitive Sciences, 4(11), 423-431. [Link](https://www.sciencedirect.com/science/article/abs/pii/S1364661300015798)

34. **Wolpert, D. M., Ghahramani, Z., & Flanagan, J. R. (2001).** Perspectives and problems in motor learning. Trends in Cognitive Sciences, 5(11), 487-494. [Link](https://www.sciencedirect.com/science/article/abs/pii/S1364661300017043)

These references provide a robust foundation for the theoretical and empirical underpinnings of the Neuromorphic Simulation Project, contributing to the development of sophisticated neural simulations and the exploration of this artificial consciousness project coded by Matthew Drabek.

*Special thanks to medical student A. Fogl


