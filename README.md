<!-- # Waterworld Policy Evolution

This project uses Genetic Algorithms and Reinforcement Learning (PPO and SAC) to evolve policies for agents in the Waterworld environment from the PettingZoo library.

## Installation

To install the necessary dependencies, run the following command:

```sh
pip install -r requirements.txt
```

## Usage

To train a model, run the `main.py` script with the `process_to_run` variable set to `'train'`. This will train the model using the specified settings and save it to the `models/train` directory.

To optimize the hyperparameters of the model, set the `process_to_run` variable to `'optimize'`. This will run the genetic algorithm to find the best hyperparameters for the model.

## Files

- `main.py`: The main script to run for training or optimizing the model.
- `ga.py`: Contains the `GeneticHyperparamOptimizer` class which is used for optimizing the hyperparameters of the model.
- `hueristic_policy.py`: Contains a basic heuristic policy for the agents in the Waterworld environment.
- `settings.py`: Contains the settings for the Waterworld environment. -->

# Waterworld w/ Communication

This project applies Genetic Algorithms, Soft Actor-Critic (SAC), and Proximal Policy Optimization (PPO) algorithms from stable-baselines3 to evolve policies for agents in the Waterworld environment, part of the PettingZoo library. It aims to enhance decision-making in swarm robotics, particularly in the context of disaster management and complex, dynamic environments.

## Installation

Before running the project, ensure that you have Python 3.8 or later installed. To install necessary dependencies, run the following command in your terminal:


```
pip install -r requirements.txt
```

## Project Structure

- `ga.py`: Implements a genetic algorithm for optimizing hyperparameters of SAC and PPO models.
- `heuristic_policy.py`: Contains a simple heuristic policy.
- `main.py`: Main script to execute the training or optimization process, integrating SAC, PPO, and genetic algorithms.
- `README.md`: This file, providing detailed information about the project.
- `settings.py`: Configuration settings for the Waterworld simulation environment.

## Usage

To run the project, execute the `main.py` script. This script provides flexibility to choose the model type and the process to run, which includes training, evaluation, or optimization of models. You can set the `mdl` variable to `"SAC"`, `"PPO"`, or `"Heuristic"` and `process_to_run` to `'train'`, `'optimize'`, or `'eval'`. Additionally, modify the `settings.py` to customize the simulation parameters and algorithm hyperparameter space configurations for tailored experimentation.

### Example:
To run a training session with the PPO model:
mdl = "PPO"  # Choose "PPO", "SAC", or "Heuristic"
process_to_run = 'train'  # Choose 'train', 'optimize', or 'eval'

To perform optimization with the SAC model:
mdl = "SAC"
process_to_run = 'optimize'

To evaluate using the heuristic policy:
mdl = "Heuristic"
process_to_run = 'eval'


## Algorithms and References

### SAC and PPO Algorithms
- SAC: An off-policy actor-critic method optimizing a trade-off between expected return and entropy. Implemented using stable-baselines3.
  - Reference: Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905.
- PPO: A policy gradient method for reinforcement learning that simplifies implementation and improves sample complexity. Implemented using stable-baselines3.
  - Reference: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

### PettingZoo Library
- A Python library for multi-agent reinforcement learning.
  - Citation: Terry, J., et al. (2021). Pettingzoo: Gym for multi-agent reinforcement learning. Advances in Neural Information Processing Systems, 34, 15032-15043.


## Contributions and Acknowledgments

This project utilizes the PettingZoo library for the Waterworld simulation environment, and the SAC and PPO algorithms from stable-baselines3 for policy development. Thanks to all contributors and maintainers of these open-source resources.



