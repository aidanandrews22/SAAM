# Example usage: python generate_dataset_cartpole_gym.py --config conf/generate_gym_data.yaml

# TODO: ignore non-stabilized trajectories
# TODO: inject suboptimal control actions

import os
import pickle
import numpy as np
from numpy._core.multiarray import dtype
import torch
from tqdm import tqdm
from quinine import QuinineArgumentParser
import yaml
import random
from concurrent.futures import ProcessPoolExecutor

# Relative imports
from controller.gym_continuous_cartpole import ContinuousCartPoleEnv
from controller.gym_cartpole_swingup_lqr import swingup_lqr_controller
from model.curriculum import Curriculum
from model.schema import schema


torch.backends.cudnn.benchmark = True


def get_valid_masses_and_lengths_uniform(
    size=1,
    cartmass=2,
    polemasslowerbound=0.3,
    polemassupperbound=1.0,
    polelengthlowerbound=1.0,
    polelengthupperbound=1.5,
):
    """
    Samples valid masses and lengths for the cartpole system using a uniform distribution.
    """
    cartmass = np.ones(size) * cartmass
    polemass = np.random.uniform(polemasslowerbound, polemassupperbound, size=size)
    polelength = np.random.uniform(
        polelengthlowerbound, polelengthupperbound, size=size
    )
    return cartmass, polemass, polelength


def generate_random_initial_conditions(size=1):
    """
    Generates random initial conditions for the cartpole system.

    Returns:
        np.ndarray: Initial conditions with shape (size, 4) - [x, theta, xdot, thetadot]
    """
    x_init = np.zeros(size)  # Cart starts at origin
    theta_init = np.random.uniform(
        np.pi - np.pi / 2, np.pi + np.pi / 2, size=size
    )  # Around inverted position
    xdot_init = np.zeros(size)  # Cart starts at rest
    thetadot_init = np.random.uniform(-1.0, 1.0, size=size)  # Random angular velocity

    return np.column_stack((x_init, theta_init, xdot_init, thetadot_init))


def run_single_trajectory(
    cartmass, polemass, polelength, initial_state, n_points, dt=0.025
):
    """
    Runs a single trajectory using the gym environment and swingup_lqr_controller.

    Args:
        cartmass: Mass of the cart
        polemass: Mass of the pole
        polelength: Length of the pole
        initial_state: Initial state [x, theta, xdot, thetadot]
        n_points: Number of timesteps to simulate
        dt: Timestep (default 0.025 to match generate_dataset_cartpole_ebonye.py)

    Returns:
        states: np.ndarray of shape (4, n_points) - [x, theta, xdot, thetadot]
        controls: np.ndarray of shape (n_points,) - control values
        modes: np.ndarray of shape (n_points,) - control modes (0=swing-up, 1=LQR)
    """
    print(f"Initial state: {initial_state}")
    env = ContinuousCartPoleEnv(
        masscart=cartmass,
        masspole=polemass,
        length=polelength,
        render_mode=None,  # No rendering for speed
    )

    # Set the timestep
    env.tau = dt

    # Reset with initial state
    obs, _ = env.reset(options={"init_state": initial_state.tolist()})

    states = [obs]
    controls = []
    modes = []
    switched = False

    for _ in range(n_points - 1):
        # Get control action from swingup_lqr_controller
        action, switched = swingup_lqr_controller(
            obs, switched, cartmass, polemass, polelength
        )

        # Store the control mode (0 for swing-up, 1 for LQR)
        mode = 1.0 if switched else 0.0

        # Step the environment
        obs, reward, done, truncated, _, applied_action = env.step(action)

        states.append(obs)
        controls.append(applied_action[0])  # Extract scalar from array
        modes.append(mode)

        if done or truncated:
            # Pad with last values if episode ends early
            while len(states) < n_points:
                states.append(obs)
            while len(controls) < n_points - 1:
                controls.append(applied_action[0])
                modes.append(mode)
            break

    env.close()

    # Convert to numpy arrays and transpose states to match expected format
    states = np.array(states)  # Shape: (n_points, 4)
    controls = np.array(controls)  # Shape: (n_points,)
    modes = np.array(modes)  # Shape: (n_points,)

    if states.shape[1] > n_points:
        states = states[:, :n_points]
    if len(controls) > n_points:
        controls = controls[:n_points]
        modes = modes[:n_points]

    return states, controls, modes


def _one_traj(args):
    return run_single_trajectory(*args)


def generate_batch_parallel(batch_size, cartmass, polemass, polelength, n_points):
    """
    Generates a batch of trajectories using parallel processing.

    Returns:
        xs: np.ndarray of shape (batch_size, n_points, 4) - states
        ys: np.ndarray of shape (batch_size, n_points-1, 1) - controls
    """
    init_states = generate_random_initial_conditions(batch_size)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as exe:
        out = list(
            exe.map(
                _one_traj,
                [
                    (cartmass[i], polemass[i], polelength[i], init_states[i], n_points)
                    for i in range(batch_size)
                ],
            )
        )
    states, controls, modes = map(list, zip(*out))
    xs = np.stack(states)  # (B, T, 4)

    # stack controls and modes
    controls_arr = np.stack(controls)
    modes_arr = np.stack(modes)
    ys = np.stack([controls_arr, modes_arr], axis=-1)  # (B, T-1, 1)
    return xs, ys


def generate_batch(
    batch_size, cartmass, polemass, polelength, n_points, device="cuda:0"
):
    """
    Generates a batch of trajectories using the gym environment.

    Returns:
        xs: torch.Tensor of shape (batch_size, 4, n_points) - states
        ys: torch.Tensor of shape (batch_size, n_points, 2) - [controls, modes]
    """
    xs_np, ys_np = generate_batch_parallel(
        batch_size, cartmass, polemass, polelength, n_points
    )
    # Convert to torch tensors - xs_np is already (batch_size, n_points, 4)
    xs = torch.tensor(
        xs_np, dtype=torch.float32, device=device
    )  # (batch_size, n_points, 4)
    ys = torch.tensor(ys_np, dtype=torch.float32, device=device)
    return xs, ys


def save_pickle(data, pickle_path):
    """Saves data to a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)


def append_to_dataset_logger(
    iteration, cartmass, polemass, polelength, xs_shape, log_file
):
    """Appends simulation details to a log file for tracking."""
    with open(log_file, "a") as f:
        f.write(f"Iteration: {iteration}\n")
        f.write("Cartmass: " + str(cartmass) + "\n")
        f.write("Polemass: " + str(polemass) + "\n")
        f.write("Polelength: " + str(polelength) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """
    Generates training datasets using the gym environment and swingup_lqr_controller.
    """
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(
        range(starting_step, args.training.train_steps + args.training.test_pendulums)
    )

    train_logger = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    test_logger = os.path.join(
        args.dataset_filesfolder, args.dataset_test_logger_textfile
    )
    base_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder)
    test_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder_test)

    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    for i in pbar:
        if i < args.training.train_steps:
            # Training data
            cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(
                size=bsize,
                cartmass=2,
                polemasslowerbound=0.3,
                polemassupperbound=1.0,
                polelengthlowerbound=1.0,
                polelengthupperbound=1.5,
            )

            # Generate batch using gym environment
            xs, control_values = generate_batch(
                bsize,
                cartmass,
                polemass,
                polelength,
                curriculum.n_points,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )

            pickle_file = f"batch_{i}.pkl"
            pickle_path = os.path.join(base_data_dir, pickle_file)

            save_pickle(
                (xs, control_values, cartmass, polemass, polelength), pickle_path
            )
            append_to_dataset_logger(
                i, cartmass, polemass, polelength, xs.shape, train_logger
            )

        elif (
            i >= args.training.train_steps
            and i < args.training.train_steps + args.training.test_pendulums
        ):
            # Test data
            cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(
                size=bsize,
                cartmass=2,
                polemasslowerbound=0.3,
                polemassupperbound=1.0,
                polelengthlowerbound=1.0,
                polelengthupperbound=1.5,
            )

            # Generate batch using gym environment
            xs, control_values = generate_batch(
                bsize,
                cartmass,
                polemass,
                polelength,
                curriculum.n_points,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
            )

            pickle_file = f"batch_test_{i - args.training.train_steps}.pkl"
            pickle_path = os.path.join(test_data_dir, pickle_file)

            save_pickle(
                (xs, control_values, cartmass, polemass, polelength), pickle_path
            )
            append_to_dataset_logger(
                i - args.training.train_steps,
                cartmass,
                polemass,
                polelength,
                xs.shape,
                test_logger,
            )

        curriculum.update()
        pbar.set_description(
            f"Generated {i + 1}/{args.training.train_steps + args.training.test_pendulums} batches"
        )


def main(args):
    global_seed = 42
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    make_train_data(args)


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    with open(os.path.join(args.dataset_filesfolder, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
