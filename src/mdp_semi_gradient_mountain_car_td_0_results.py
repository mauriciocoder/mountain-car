import argparse
import csv
import glob
import logging
import math
import pickle
import random
import statistics
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
            "Proto Points": self.protos_per_dimension,
            "Training Session": training_session,
            "Simulation": simulation,
            "Trajectory Size": trajectory_size,
            # "Trajectory": "" if trajectory_size > 1000 else trajectory,
            # "Weights": "" if trajectory_size > 1000 else self.weights
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


def write_model_convergence_to_csv(
    model_convergence: dict, filename: str, all_discounting: bool = True
) -> None:
    for key, value in model_convergence.items():
        if all_discounting:
            feature_type, alpha, gamma, polynomial_dimension, protos_per_dimension, epsilon = key.split(
                ";"
            )
            row = {
                "Feature Type": feature_type,
                "Alpha": alpha,
                "Gamma": gamma,
                "Polynomial Dimension": polynomial_dimension,
                "Protos per dimension": protos_per_dimension,
                "Epsilon": epsilon,
                "Convergent Training Sessions (from 10 training sessions)": str(value)
            }
        else:
            polynomial_dimension, protos_per_dimension = key.split(
                ";"
            )
            row = {
                "Polynomial Dimension": polynomial_dimension,
                "Protos per dimension": protos_per_dimension,
                "Convergent Training Sessions": str(value)
            }
        logging.info(f"Writing row: {row}")
        # Open the CSV file in append mode
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            # Write header only if the file is empty
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)


def write_model_simluation_to_csv(
    mc: MountainCar, trajectory_sizes: list[int], filename: str
) -> None:
    min_size = min(trajectory_sizes)
    max_size = max(trajectory_sizes)
    row = {
        "Feature Type": mc.feature_type,
        "Alpha": mc.alpha,
        "Gamma": mc.gamma,
        "Polynomial Dimension": mc.polynomial_dimension,
        "Protos per dimension": mc.protos_per_dimension,
        "Epsilon": mc.epsilon,
        "Simulations": len(trajectory_sizes),
        "MIN": min_size,
        "MAX": max_size,
        "Delta": max_size - min_size,
        "Mean": statistics.mean(trajectory_sizes),
        "STDEV": statistics.stdev(trajectory_sizes)
    }
    logging.info(f"Writing row: {row}")
    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        # Write header only if the file is empty
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def simulate_from_saved_models(
    trained_models_dir: str, feature_type: str, model_simulation_results_filename: str,
    model_convergence_all_discounting_results_filename: str,
    model_convergence_main_discounting_results_filename: str
) -> None:
    try:
        file_extension = "*.pkl"
        file_paths = glob.glob(trained_models_dir + file_extension)
        # Write training session convergence
        model_convergence_by_all_discountings = {}
        model_convergence_by_main_discounting = {}
        for file_path in file_paths:
            mc = MountainCar.load(file_path, feature_type)

            key = f"{mc.feature_type};{mc.alpha};{mc.gamma};{mc.polynomial_dimension};{mc.protos_per_dimension};{mc.epsilon}"
            model_convergence_by_all_discountings.setdefault(key, 0)
            model_convergence_by_all_discountings[key] += 1

            key = f"{mc.polynomial_dimension};{mc.protos_per_dimension}"
            model_convergence_by_main_discounting.setdefault(key, 0)
            model_convergence_by_main_discounting[key] += 1

            # Write simulation results
            # Mean of steps, standard deviation, min, max
            trajectory_sizes = []
            for _ in range(1000):
                trajectory = mc.simulate(random_initial_pos=True)
                trajectory_size = len(trajectory)
                if trajectory_size >= 5000:
                    # Only persist models which is convergent on every simulation
                    continue
                trajectory_sizes.append(trajectory_size)
            write_model_simluation_to_csv(
                mc, trajectory_sizes, model_simulation_results_filename
            )
        write_model_convergence_to_csv(
            model_convergence_by_all_discountings,
            model_convergence_all_discounting_results_filename
        )
        write_model_convergence_to_csv(
            model_convergence_by_main_discounting,
            model_convergence_main_discounting_results_filename, all_discounting=False
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logging.info("End of program")

# $ python3 mdp_semi_gradient_mountain_car_td_0_results.py &
if __name__ == '__main__':
    simulate_from_saved_models(
        "/home/mauricio/dev/source/play/trained_models/polynomial/", "POLYNOMIAL",
        "polynomial_model_simulation_results.csv",
        "polynomial_model_convergence_all_discounting_results.csv",
        "polynomial_model_convergence_main_discounting_results.csv"
    )

    simulate_from_saved_models(
        "/home/mauricio/dev/source/play/trained_models/nrb/", "NRB",
        "nrb_model_simulation_results.csv",
        "nrb_model_convergence_all_discounting_results.csv",
        "nrb_model_convergence_main_discounting_results.csv",
    )