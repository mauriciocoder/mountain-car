# Temporal Difference Learning - TD(0) - for Mountain Car Problem

Temporal Difference TD(0) learning has been utilized to develop robust evaluation functions across a variety of scenarios. In this algorithm we expose a class called MountainCar that implements two different feature selection methods, radial basis functions (RBF) and polynomial feature selections, in the context of episodic semi-gradient TD(0). Our evaluation focuses on solving the well-studied "mountain-car" problem, which challenges the model with a large and continuous input space.

## Requirements

- Python version 3.10 with `pip` installed.

## Installation

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Polynomial Feature Version

You can run the polynomial version using the following command:

```bash
python3 src/mdp_semi_gradient_mountain_car_td_0.py --feature_type POLYNOMIAL --alpha_list 0.010 0.025 0.05 0.075 0.1 0.125 0.150 0.175 0.2 0.4 0.5 --gamma_list 0.8 0.9 0.95 0.99 1.0 --epsilon 0.25 0.5 0.75 --polynomial_dimension_list 2 3 4 5 6 7 8 9 10 --training_sessions 10 --simulations 100
```

You can run the RBF version using the following command:

```bash
python3 src/mdp_semi_gradient_mountain_car_td_0.py --feature_type NRB --alpha_list 0.010 0.025 0.05 0.075 0.1 0.125 0.150 0.175 0.2 0.4 0.5 --gamma_list 0.8 0.9 0.95 0.99 1.0 --epsilon 0.25 0.5 0.75 --protos_per_dimension_list 8 16 32 64 128 --training_sessions 10 --simulations 100
```

**Parameters:**
- `--feature_type`: Feature selection method (`POLYNOMIAL` for polynomial features or `NRB` for Radial Basis Function).
- `--alpha_list`: Learning rates for training sessions.
- `--gamma_list`: Discount factors for the reward.
- `--epsilon`: Exploration parameter for epsilon-greedy strategy.
- `--polynomial_dimension_list`: List of polynomial dimensions.
- `--protos_per_dimension_list`: Number of prototypes per dimension for RBF.
- `--training_sessions`: Number of training sessions.
- `--simulations`: Number of simulations.

Make sure to adjust the parameters based on your experiment requirements. The provided commands serve as examples for running the experiments with different configurations.
