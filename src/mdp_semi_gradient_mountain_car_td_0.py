"""
Richard S. Sutton_ Andrew G Barto - Reinforcement Learning_ An Introduction-Bradford Book (2018)

Example 10.1: Mountain Car Task - Page 245

Consider the task of driving an underpowered
car up a steep mountain road, as suggested by the diagram in the upper left of Figure 10.1.
The difficulty is that gravity is stronger than the car’s engine, and even at full throttle
the car cannot accelerate up the steep slope. The only solution is to first move away from
the goal and up the opposite slope on the left. Then, by applying full throttle the car
can build up enough inertia to carry it up the steep slope even though it is slowing down
the whole way. This is a simple example of a continuous control task where things have
to get worse in a sense (farther from the goal) before they can get better. Many control
methodologies have great difficulties with tasks of this kind unless explicitly aided by a
human designer.
The reward in this problem is 1 on all time steps until the car moves past its goal
position at the top of the mountain, which ends the episode. There are three possible
actions: full throttle forward (+1), full throttle reverse ( 1), and zero throttle (0). The
car moves according to a simplified physics. Its position, xt , and velocity, ẋt , are updated
by
xt+1 = bound xt + ẋt+1
ẋt+1 = bound ẋt + 0.001At - 0.0025 * cos(3xt ) ,
where the bound operation enforces 1.2  xt+1  0.5 and 0.07  ẋt+1  0.07. In
addition, when xt+1 reached the left bound, ẋt+1 was reset to zero. When it reached
the right bound, the goal was reached and the episode was terminated. Each episode
started from a random position xt 2 [ 0.6, 0.4) and zero velocity.

This implementation is applying the Episodic Semi-gradient Sarsa for Estimating q̂ from page 244.
The features selected are based on the normalized basis vector extracted from the 2d-space of
position and velocity. We have created 35x35 central points in the plane such that
each pair (position, velocity) has a relative distance to each of the 35x35 central points. The nearest
points will have highere value and the total sum of distances for a certain pair should sum 1.

Implementation based on explanation: https://youtu.be/Vky0WVh_FSk?si=-WLDsT2iZW40ncw1&t=866

"""
import argparse
import csv
import logging
import math
import pickle
import random
from datetime import datetime
from functools import cache
from itertools import product
from typing import Final, Union

import numpy as np
from attrs import define, field, validators
from numpy.polynomial.polynomial import polyvander2d

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f'mountain-car_{timestamp}.log'
logging.basicConfig(
    filename=log_filename, level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

POS_LIMITS: tuple[float, float] = (-1.2, 0.5)
VEL_LIMITS: tuple[float, float] = (-0.07, 0.07)

FULL_THROTTLE_FORWARD: Final = 0
FULL_THROTTLE_REVERSE: Final = 1
ZERO_THROTTLE: Final = 2


@define
class MountainCar:
    feature_type: str = field(
        validator=validators.in_(("POLYNOMIAL", "NRB"))
    )  # Selector for feature generation
    alpha: float  # Learning rate
    gamma: float  # Next step weight
    epsilon: float  # Used in e-greedy evaluation (exploration x exploitation)
    weights: np.array = None
    episodes: int = 500  # Episodes used for training
    protos_per_dimension: int = 35  # Points used per dimension, used for NRB
    polynomial_dimension: int = 2  # Dimensions for polynomial
    _centers: np.array = field(init=False)  # Centers used for NRB

    def __attrs_post_init__(self):
        if self.feature_type == "NRB":
            proto_pos = np.linspace(*POS_LIMITS, self.protos_per_dimension + 2)[1:-1]
            proto_vel = np.linspace(*VEL_LIMITS, self.protos_per_dimension + 2)[1:-1]
            self._centers = np.array(list(product(proto_pos, proto_vel)))
            weights_dimensions = self._centers.shape[0]
        else:
            weights_dimensions = int(math.pow(self.polynomial_dimension + 1, 2))
        if self.weights is None:
            self.weights = [np.zeros(weights_dimensions) for _ in
                range(3)]  # One weight vector for each action

    def train(self) -> None:
        """
        Train the model by updating the weights. If model does not converge to final state,
        skip the episodes.
        """
        pos = random.uniform(-0.6, -0.4)
        vel = 0.0
        action = None
        logging.info(
            f"Trainining for alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, polynomial_dimension={self.polynomial_dimension}, protos_per_dimension={self.protos_per_dimension}"
        )
        logging.info(f"Initial state={(pos, vel)}")
        for _ in range(self.episodes):
            state = (pos, vel)
            counter = 0
            features = self._get_features(state)
            while True:
                counter += 1
                action, reward, next_state = self.take_action(
                    action, state
                )
                if MountainCar.is_final_state(next_state) or counter > 5000:
                    self._update_weights(features, action, reward)
                    break
                next_action, _, _ = self.take_action(None, next_state)
                next_features = self._get_features(next_state)
                self._update_weights(
                    features, action, reward, next_features, next_action
                )
                state = next_state
                action = next_action
                features = next_features

    def simulate(self, random_initial_pos: bool = False) -> list[tuple[float, float]]:
        """
        Given trained weights, simulate the car movement and return its trajectory.
        """
        logging.info(
            f"Simulating for alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}, polynomial_dimension={self.polynomial_dimension}, randomInitialPos={random_initial_pos}"
        )
        pos = random.uniform(-0.6, -0.4) if random_initial_pos else -0.5
        vel = 0.0
        state = (pos, vel)
        logging.info(f"Initial state={state}")
        counter = 0
        trajectory = []
        while not MountainCar.is_final_state(state) and counter < 5000:
            features = self._get_features(state)
            best_action = self._get_best_action(features)
            action, reward, next_state = self.take_action(best_action, state)
            # print(f"State: {state}, action: {action}")
            trajectory.append(state)
            state = next_state
            counter += 1
        max_tuple = max(trajectory, key=lambda state: state[0])
        trajectory_size = len(trajectory)
        logging.info(
            f"The max position reached is in state: {max_tuple}. Total trajectory size: {trajectory_size}"
        )
        return trajectory

    def write_results_to_csv(
        self, trajectory: list[tuple[float, float]],
        filename: str, training_session: int = 0, simulation: int = 0
    ) -> None:
        trajectory_size = len(trajectory)

        row = {
            "Alpha": self.alpha, "Gamma": self.gamma, "Epsilon": self.epsilon,
            "Polynomial Dimension": self.polynomial_dimension,
            "Training Session": training_session,
            "Simulation": simulation,
            "Trajectory Size": trajectory_size,
            "Trajectory": "" if trajectory_size > 1000 else trajectory,
            "Weights": "" if trajectory_size > 1000 else self.weights
        }
        # Open the CSV file in append mode
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            # Write header only if the file is empty
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def serialize_trained_data(
        self, training_session: int = 0, trajectory_sizes_mean: float = 0.0
    ) -> None:
        filename = f"alpha_{self.alpha}_gamma_{self.gamma}_epsilon_{self.epsilon}_polynomial_dimension_{self.polynomial_dimension}_proto_points_{self.protos_per_dimension}_training_session_{training_session}.pkl"
        trained_data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "polynomial_dimension": self.polynomial_dimension,
            "protos_per_dimension": self.protos_per_dimension,
            "trajectory_sizes_mean": trajectory_sizes_mean,
            "weights": self.weights
        }
        with open(filename, 'wb') as file:
            pickle.dump(trained_data, file)

    @staticmethod
    def load(file_path: str, feature_type: str) -> "MountainCar":
        with open(file_path, 'rb') as file:
            trained_data = pickle.load(file)
            if feature_type == "NRB":
                return MountainCar(
                    feature_type=feature_type, alpha=trained_data["alpha"],
                    gamma=trained_data["gamma"],
                    epsilon=trained_data["epsilon"],
                    protos_per_dimension=trained_data["protos_per_dimension"],
                    weights=trained_data["weights"]
                )
            elif feature_type == "POLYNOMIAL":
                return MountainCar(
                    feature_type=feature_type, alpha=trained_data["alpha"],
                    gamma=trained_data["gamma"],
                    epsilon=trained_data["epsilon"],
                    polynomial_dimension=trained_data["polynomial_dimension"],
                    weights=trained_data["weights"]
                )
            raise TypeError("Unable to load serialized object")

    def take_action(self, action: int | None, state: tuple[float, float]) -> tuple[
        int, int, tuple[float, float]]:
        """
        Take an action, if no action is provided finds an action based on e-greedy policy.
        """
        if action is None:
            # Choose an action
            choices = ["BEST", "RANDOM"]
            selected = \
                random.choices(choices, weights=[1 - self.epsilon, self.epsilon], k=1)[
                    0]
            if selected == "RANDOM":
                action = random.choice(
                    [FULL_THROTTLE_FORWARD, FULL_THROTTLE_REVERSE, ZERO_THROTTLE]
                )
            else:
                features = self._get_features(state)
                action = self._get_best_action(features)
        next_state = MountainCar.move_car(state, action)
        reward = -1 if next_state[0] < POS_LIMITS[1] else 0
        return action, reward, next_state

    @staticmethod
    def is_final_state(state: tuple) -> bool:
        return state[0] >= POS_LIMITS[1]

    @staticmethod
    def move_car(state: tuple, action: int) -> tuple[float, float]:
        """
        Apply given action to car and returns its new state (position, velocity).
        """
        acceleration = 1 if action == FULL_THROTTLE_FORWARD else -1 if action == FULL_THROTTLE_REVERSE else 0
        next_vel = state[1] + 0.001 * acceleration - 0.0025 * math.cos(3 * state[0])
        next_vel = max(VEL_LIMITS[0], min(next_vel, VEL_LIMITS[1]))
        next_pos = state[0] + next_vel
        next_pos = max(POS_LIMITS[0], min(next_pos, POS_LIMITS[1]))
        return next_pos, next_vel

    @staticmethod
    @cache
    def get_scaler(distance_scaler: float = 0.005) -> np.array:
        vel_scale = VEL_LIMITS[1] - VEL_LIMITS[0]
        pos_scale = POS_LIMITS[1] - POS_LIMITS[0]
        return np.array([pos_scale, vel_scale]) * distance_scaler

    def _get_radial_basis_features(
        self, state: tuple[float, float], distance_scaler: float = 0.005
    ) -> np.array:
        """
        Returns the Normalized Basis Vector (Gaussian) for a given state.
        Reference: https://en.wikipedia.org/wiki/Radial_basis_function
        """
        state_array = np.array(state)
        scaler = MountainCar.get_scaler(distance_scaler)
        dist_components = (state_array - self._centers) / scaler
        radial_basis = np.exp(-(dist_components ** 2).sum(1))
        return radial_basis / radial_basis.sum()

    def _get_polynomial_features(self, state: tuple[float, float]) -> np.array:
        """
        Returns the 2D polynomial vector based on dimension
        """
        pos = state[0]
        vel = state[1]
        vel_scale = VEL_LIMITS[1] - VEL_LIMITS[0]
        pos_scale = POS_LIMITS[1] - POS_LIMITS[0]
        # Normalization based on Min-Max scaling
        normalized_pos = (pos - POS_LIMITS[0]) / pos_scale
        normalized_vel = (vel - VEL_LIMITS[0]) / vel_scale

        # Create the 2D Vandermonde matrix
        vander_matrix = polyvander2d(
            normalized_pos, normalized_vel,
            [self.polynomial_dimension, self.polynomial_dimension]
        )
        # Flatten the Vandermonde matrix to get the coefficients
        coefficients = vander_matrix.flatten()
        return coefficients

    def _get_features(self, state: tuple[float, float]) -> np.array:
        if self.feature_type == "NRB":
            return self._get_radial_basis_features(state)
        elif self.feature_type == "POLYNOMIAL":
            return self._get_polynomial_features(state)
        else:
            raise ValueError("Unsuported operation")

    def _get_best_action(self, features: np.array) -> int:
        """
        Best action is the one with the highest q-value over each of the 3 actions.
        """
        forward: float = np.dot(self.weights[FULL_THROTTLE_FORWARD], features)
        reverse: float = np.dot(self.weights[FULL_THROTTLE_REVERSE], features)
        zero: float = np.dot(self.weights[ZERO_THROTTLE], features)
        best: float = max(forward, reverse, zero)
        if best == forward:
            return FULL_THROTTLE_FORWARD
        if best == reverse:
            return FULL_THROTTLE_REVERSE
        return ZERO_THROTTLE

    def _update_weights(
        self, features: np.array, action: int, reward: int,
        next_features: Union[np.array, None] = None, next_action: int | None = None
    ):
        """
        Episodic Semi-gradient Sarsa for Estimating q_star. Reference: Page 244
        Update weights used for learning.
        """
        q = np.dot(self.weights[action], features)
        if next_features is None:
            # Final state
            self.weights[action] += self.alpha * (reward - q) * features
        else:
            # One-step look ahead
            next_q = np.dot(self.weights[next_action], next_features)
            self.weights[action] += self.alpha * (
                reward + (self.gamma * next_q) - q) * features


def get_args() -> any:
    parser = argparse.ArgumentParser(description="Example CLI program")
    parser.add_argument(
        "--feature_type", type=str,
        help="Feature Type: POLYNOMIAL or NRB"
    )
    parser.add_argument(
        "--alpha_list", nargs="+", type=float, help="List of alpha for training"
    )
    parser.add_argument(
        "--gamma_list", nargs="+", type=float, help="List of gamma for training"
    )
    parser.add_argument(
        "--epsilon_list", nargs="+", type=float, help="List of epsilon for training"
    )
    parser.add_argument(
        "--polynomial_dimension_list", nargs="+", type=int,
        help="List of polynomial_dimensions for training"
    )
    parser.add_argument(
        "--protos_per_dimension_list", nargs="+", type=int,
        help="List of protos_per_dimension for training"
    )
    parser.add_argument(
        "--training_sessions", type=int,
        help="Number of training sessionss for execution"
    )
    parser.add_argument(
        "--simulations", type=int,
        help="Number of simulations to run on trained data"
    )
    args = parser.parse_args()
    if args.polynomial_dimension_list is None:
        args.polynomial_dimension_list = [1]
    if args.protos_per_dimension_list is None:
        args.protos_per_dimension_list = [1]
    return args


# This is the main function running in the servers codecraft-instance-01 and codecraft-instance-02
# Execution example:
# Polynomial:
# $ python3 mdp_semi_gradient_mountain_car_td_0.py --feature_type POLYNOMIAL --alpha_list 0.010 0.025 0.05 0.075 0.1 0.125 0.150 0.175 0.2 0.4 0.5 --gamma_list 0.8 0.9 0.95 0.99 1.0 --epsilon 0.25 0.5 0.75 --polynomial_dimension_list 2 3 4 5 6 7 8 9 10 --training_sessions 10 --simulations 100
# NRB:
# $ python3 mdp_semi_gradient_mountain_car_td_0.py --feature_type NRB --alpha_list 0.010 0.025 0.05 0.075 0.1 0.125 0.150 0.175 0.2 0.4 0.5 --gamma_list 0.8 0.9 0.95 0.99 1.0 --epsilon 0.25 0.5 0.75 --protos_per_dimension_list 8 16 32 64 128 --training_sessions 10 --simulations 100
if __name__ == '__main__':
    args = get_args()
    logging.info(f"Starting training with params: {args}")
    training_sessions = args.training_sessions
    simulations = args.simulations
    try:
        for alpha in args.alpha_list:
            for gamma in args.gamma_list:
                for epsilon in args.epsilon_list:
                    for polynomial_dimension in args.polynomial_dimension_list:
                        for protos_per_dimension in args.protos_per_dimension_list:
                            trajectory_sizes_means = []
                            for training_session in range(training_sessions):
                                logging.info(f"Training Session #{training_session}")
                                mc1 = MountainCar(
                                    feature_type=args.feature_type, alpha=alpha,
                                    gamma=gamma,
                                    epsilon=epsilon,
                                    polynomial_dimension=polynomial_dimension,
                                    protos_per_dimension=protos_per_dimension
                                )
                                mc1.train()
                                trajectory_sizes = []
                                for simulation in range(100):
                                    logging.info(f"Simulation #{simulation}")
                                    trajectory = mc1.simulate(random_initial_pos=True)
                                    trajectory_sizes.append(len(trajectory))
                                    mc1.write_results_to_csv(
                                        trajectory, "trajectory_results.csv",
                                        training_session, simulation
                                    )
                                trajectory_sizes_mean = np.mean(trajectory_sizes)
                                if trajectory_sizes_mean <= 300:
                                    logging.info(
                                        f"Serializing trained data for session #{training_session}"
                                    )
                                    mc1.serialize_trained_data(
                                        training_session=training_session,
                                        trajectory_sizes_mean=trajectory_sizes_mean
                                    )
                                trajectory_sizes_means.append(np.mean(trajectory_sizes))
                            logging.info(
                                f"Trajectory size means after {training_sessions} training sessions for alpha={alpha}, gamma={gamma}, epsilon={epsilon}, polynomial_dimension={polynomial_dimension} -> {trajectory_sizes_means}"
                            )
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logging.info("End of program")

"""
if __name__ == '__main__':
    try:
        mc = MountainCar.load("/home/mauricio/dev/source/play/src/alpha_0.01_gamma_0.8_epsilon_0.25_polynomial_dimension_1_proto_points_8_training_session_3.pkl", "NRB")
        for simulation in range(100):
            logging.info(f"Simulation #{simulation}")
            trajectory = mc.simulate(random_initial_pos=True)
            size = len(trajectory)
            logging.info(f"Simulation #{simulation} size: {size}")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logging.info("End of program")
"""