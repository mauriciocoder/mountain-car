# Temporal Difference Learning - TD(0) - for Mountain Car Problem

Temporal Difference TD(0) learning has been utilized to develop robust evaluation functions across a variety of scenarios. In this algorithm we expose a class called MountainCar that implements two different feature selection methods, radial basis functions (NRB) and polynomial feature selections, in the context of episodic semi-gradient TD(0). Our evaluation focuses on solving the well-studied "mountain-car" problem, which challenges the model with a large and continuous input space.

## Requirements

- Python version 3.10 with `pip` installed.

## Installation

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Training the Models

### Polynomial Selection Feature

You can run the polynomial version using the following command:

```bash
python3 src/mdp_semi_gradient_mountain_car_td_0.py --feature_type POLYNOMIAL --alpha_list 0.010 0.025 0.05 0.075 0.1 0.125 0.150 0.175 0.2 0.4 0.5 --gamma_list 0.8 0.9 0.95 0.99 1.0 --epsilon 0.25 0.5 0.75 --polynomial_dimension_list 2 3 4 5 6 7 8 9 10 --training_sessions 10 --simulations 100
```

### NRB Selection Feature

You can run the NRB version using the following command:

```bash
python3 src/mdp_semi_gradient_mountain_car_td_0.py --feature_type NRB --alpha_list 0.010 0.025 0.05 0.075 0.1 0.125 0.150 0.175 0.2 0.4 0.5 --gamma_list 0.8 0.9 0.95 0.99 1.0 --epsilon 0.25 0.5 0.75 --protos_per_dimension_list 8 16 32 64 128 --training_sessions 10 --simulations 100
```

**Parameters:**
- `--feature_type`: Feature selection method (`POLYNOMIAL` for polynomial features or `NRB` for Radial Basis Function).
- `--alpha_list`: Learning rates for training sessions.
- `--gamma_list`: Discount factors for the reward.
- `--epsilon`: Exploration parameter for epsilon-greedy strategy.
- `--polynomial_dimension_list`: List of polynomial dimensions.
- `--protos_per_dimension_list`: Number of prototypes per dimension for NRB.
- `--training_sessions`: Number of training sessions.
- `--simulations`: Number of simulations.

Make sure to adjust the parameters based on your experiment requirements. The provided commands serve as examples for running the experiments with different configurations.


## Simulating the models

Once the models are trained, they are saved in files with extension *.pkl. These files can be used to simulate the mountain-car problem under different postitions and velocities. You can run the simulation using the following command:

```bash
$ python3 mdp_semi_gradient_mountain_car_td_0_results.py &
```

You should parameterize the folder directories in the `mdp_semi_gradient_mountain_car_td_0_results.py` main function to read the Polynomial and NRB feature selected models. We have trained the models previously and versioned these files in the `trained-models` folder. 


**Report generation:**

The routine will simulate the problem 1000 times. This will generate the following reports:

- `polynomial_model_simulation_results.csv`:
- `polynomial_model_convergence_all_discounting_results.csv`:
- `polynomial_model_convergence_main_discounting_results.csv`:
- `nrb_model_simulation_results.csv`:
- `nrb_model_convergence_all_discounting_results.csv`:
- `nrb_model_convergence_main_discounting_results.csv`:

The `*_model_simulation_results.csv` files expose the statistics (mean, stdev, min, max) for each trained model that was simulated 1000 times.

The `*_model_convergence_all_discounting_results.csv` expose the training sessions required (from 10 training sessions) to reach convergence for each set of discountings and feature set applied.

The `*_model_convergence_main_discounting_results.csv` expose the sum of training sessions that reached convergence for the parameters applied (for NRB the proto points per dimension in {8, 12, 18, 21, 27, 36, 52}; for polynomial we have polynomial dimension in range [2, 10]).

We have simulated the models previously and versioned these files in the `simulation-results` folder. 