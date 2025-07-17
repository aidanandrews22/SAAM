import os
import random
from tqdm import tqdm
from samplers import PendulumSampler
from samplers import CartPoleSampler
from samplers import CartPoleSampler2
from aidan_transformer_control._model.curriculum import Curriculum
from random import randint
import uuid
import ipdb

from quinine import QuinineArgumentParser
import torch
import yaml
from aidan_transformer_control._model.schema import schema
from aidan_transformer_control._model.models import build_model
import math
import random
import numpy as np
import torch
import pickle
import crocoddyl

# seed = [1]
torch.backends.cudnn.benchmark = True

def get_valid_masses_and_lengths_uniform(size=1, cartmass= 2, polemasslowerbound=0.3, polemassupperbound=1.0, polelengthlowerbound=1.0, polelengthupperbound=1.5, dt = 0.01):
    """
    Samples valid masses and lengths for a pendulum system that meet specific constraints, using a uniform distribution.

    Args:
        cartmasslowerbound (float, optional): The lower bound for the cart mass value. Defaults to 2.
        cartmassupperbound (float, optional): The upper bound for the cart mass value. Defaults to 3.
        polemasslowerbound (float, optional): The lower bound for the pole mass value. Defaults to 0.5.
        polemassupperbound (float, optional): The upper bound for the pole mass value. Defaults to 1.5.
        polelengthlowerbound (float, optional): The lower bound for the pole length value. Defaults to 0.5.
        polelengthupperbound (float, optional): The upper bound for the pole length value. Defaults to 1.5.
        dt (float, optional): The time step for the simulation. Defaults to 0.025.

    Returns:
        tuple: A tuple containing:
            - cartmass (float): The sampled cart mass value.
            - polemass (float): The sampled pole mass value.
            - polelength (float): The sampled pole length value.
    """
    # while True:
    # masses = sample_mass_uniform()
    # lengths = sample_length_uniform()
    # if is_valid_mass_length(masses, lengths, dt=0.01):
    #     return masses, lengths

    # cartmass = sample_mass_uniform(cartmasslowerbound, cartmassupperbound)
    # cartmass = cartmass
    cartmass = np.ones(size) * cartmass  # Ensure cartmass is a tensor of the specified size
    polemass = sample_mass_uniform(polemasslowerbound, polemassupperbound, size=size)
    polelength = sample_length_uniform(polelengthlowerbound, polelengthupperbound, size=size)
    # if is_valid_mass_length(dt, cartmass, polemass, polelength):
    return cartmass, polemass, polelength


def sample_mass_uniform(lower_bound=0.06, upper_bound=1.06, size=1):
    # before: 0.08, 0.12 #### 2/24/2025 (ebonye) make wider range
    """
    Samples a mass value from a uniform distribution within specified bounds.

    Args:
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.

    Returns:
        float: A sampled mass value within the specified bounds.
    """
    value = np.random.uniform(lower_bound, upper_bound, size=size)
    # seed[0] += 1
    # reseed_all(seed[0])
    return value

def sample_length_uniform(lower_bound=0.06, upper_bound=1.06, size=1):
    #before: lower_bound=0.25, upper_bound=0.45 #### 2/24/2025 (ebonye) make wider range
    """
    Samples a length value from a uniform distribution within specified bounds.

    Args:
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.

    Returns:
        float: A sampled length value within the specified bounds.
    """
    value = np.random.uniform(lower_bound, upper_bound, size=size)
    # seed[0] += 1
    # reseed_all(seed[0])
    return value


# def is_valid_mass_length(mass, length, dt):
#     """
#     Validates mass and length values to ensure they do not cause issues with rk4 during simulation.

#     Args:
#         mass (float): The mass of the pendulum.
#         length (float): The length of the pendulum.
#         dt (float): The time step for the simulation.

#     Returns:
#         bool: True if the mass and length values are valid, False otherwise.
#     """
#     g = 9.81 
#     natural_frequency = math.sqrt(g / length)  
#     moment_of_inertia = mass * length**2
#     if moment_of_inertia < 1e-6: 
#         return False
#     if natural_frequency * dt > 0.1: 
#         return False
#     return True
def is_valid_mass_length(dt, cartmass, polemass, polelength):
    """
    Validates mass and length values to ensure they do not cause issues with rk4 during simulation.

    Args:
        cartmass (float): The mass of the cart.
        polemass (float): The mass of the pole.
        polelength (float): The length of the pole.

    Returns:
        bool: True if the mass and length values are valid, False otherwise.
    """
    g = 9.81 

    if cartmass <= 0 or polemass <= 0 or polelength <= 0:
        return False
    
    moment_of_inertia = (cartmass + polemass) * polelength**2
    if moment_of_inertia < 1e-6:
        return False
    
    omega = math.sqrt(g / polelength)

    if omega * dt > 0.25:
        return False
    
    if polemass/cartmass > 1.5:
        return False

    return True
    


def generate_random_X0(theta_range=(-np.pi, np.pi), thetadot_range=(-3, 3)):
    """
    Generates a random initial state for a pendulum system.

    Returns:
        list: A list containing:
            - theta (float): A randomly generated theta, sampled uniformly from range [-π, π].
            - thetadot (float): A randomly generated thetadot, sampled uniformly from the range [-10, 10].
    """
    theta = np.random.uniform(*theta_range)
    thetadot = np.random.uniform(*thetadot_range)
    return [theta, thetadot]

def save_pickle(data, pickle_path):
    """
    Saves data to a pickle file.

    Args:
        data (any): The data to save.
        pickle_path (str): The path to the pickle file where the data will be saved.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


# def append_to_seed_file(seed_file, iteration, seed, masses, lengths, k_values, xs_shape):
#     """
#     Appends simulation details to a seed file for tracking and reproducibility.

#     Args:
#         seed_file (str): The path to the seed file.
#         iteration (int): The iteration number of the simulation.
#         seed (int): The seed value used for the simulation.
#         masses (list[float]): The mass used in the simulation.
#         lengths (list[float]): The length used in the simulation.
#         k_values (list[float]): The gain matrix used in the simulation.
#         xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
#     """
#     with open(seed_file, 'a') as f:
#         f.write(f"Iteration: {iteration}\n")
#         f.write(f"Seed: {seed}\n")
#         f.write("Masses: " + str(masses) + "\n")
#         f.write("Lengths: " + str(lengths) + "\n")
#         f.write("K: " + str(k_values) + "\n")
#         f.write(f"xs.shape: {xs_shape}\n")
#         f.write("-" * 10 + "\n")

# def append_to_dataset_logger(iteration, masses, lengths, xs_shape, log_file):
#     """
#     Appends simulation details to a seed file for tracking and reproducibility.

#     Args:
#         iteration (int): The iteration number of the simulation.
#         masses (list[float]): The mass used in the
#         lengths (list[float]): The length used in the simulation.
#         xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
#         log_file (str): The path to the log file.
#     """
#     with open(log_file, 'a') as f:
#         f.write(f"Iteration: {iteration}\n")
#         f.write("Masses: " + str(masses) + "\n")
#         f.write("Lengths: " + str(lengths) + "\n")
#         f.write(f"xs.shape: {xs_shape}\n")
#         f.write("-" * 10 + "\n")

def append_to_dataset_logger(iteration, cartmass, polemass, polelength, xs_shape, log_file):
    """
    Appends simulation details to a seed file for tracking and reproducibility.
    
    Args:
        iteration (int): The iteration number of the simulation.
        cartmass (float): The mass of the cart used in the simulation.
        polemass (float): The mass of the pole used in the simulation.
        polelength (float): The length of the pole used in the simulation.
        xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
        log_file (str): The path to the log file.
    """
    with open(log_file, 'a') as f:
        f.write(f"Iteration: {iteration}\n")
        f.write("Cartmass: " + str(cartmass) + "\n")
        f.write("Polemass: " + str(polemass) + "\n")
        f.write("Polelength: " + str(polelength) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """
    Geneates training datasets for an inverted pendulum system simulation and saves them 
    as pickle files along with metadata for reproducibility.

    Args:
        args (Namespace):
            - args.training.train_steps (int): The total number of pickle files.
            - args.training.batch_size (int): batch of each pickle file.
            - args.training.curriculum (dict): Curriculum settings to adjust training parameters dynamically.
            - args.dataset_filesfolder (str): The directory where dataset files and logs are stored.
            - args.dataset_logger_textfile (str): The name of the file for logging dataset info.
            - args.pickle_folder (str): The dataset_filesfolder subfolder where generated dataset pickle files are saved.

    Notes:
        - The `PendulumSampler` is used to generate the dataset based on random valid pendulum parameters (masses and lengths).
        - The generated datasets are saved as pickle files named in the format `multipendulum_{i}.pkl`.
        - Metadata such as the current seed, pendulum parameters, and dataset shape is logged in a separate file for reproducibility.
        - The curriculum dynamically updates training parameters, such as the number of points in each dataset.

    """
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps + args.training.test_pendulums + args.training.test_pendulums_outofdistr)) 
    # pbar_test = tqdm(range(args.training.test_pendulums))
    # num_test_pendulums = args.training.test_pendulums

    # seed_file = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    train_logger = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    test_logger = os.path.join(args.dataset_filesfolder, args.dataset_test_logger_textfile)
    # test_logger_outofdistr = os.path.join(args.dataset_filesfolder, args.dataset_test_outofdistr_logger_textfile) ## 3/5/2025 out of distribution data
    base_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder)
    test_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder_test)
    # test_data_dir_outofdistr = os.path.join(args.dataset_filesfolder, args.pickle_folder_test_outofdistr) ## 3/5/2025 out of distribution data
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    # os.makedirs(test_data_dir_outofdistr, exist_ok=True) ## 3/5/2025 out of distribution data

   
    for i in pbar:
          
        if i < args.training.train_steps:
            cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(size=bsize, cartmass=2, polemasslowerbound=0.3, polemassupperbound=1.0, polelengthlowerbound=1.0, polelengthupperbound=1.5) 
            sampler = CartPoleSampler2(n_dims=4)
            T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, bsize = bsize, cartmass=cartmass, polemass=polemass, polelength=polelength) 
            
            pickle_file = f'batch_{i}.pkl'
            pickle_path = os.path.join(base_data_dir, pickle_file)

            # save_pickle((xs, control_values, cartmass, polemass, polelength), pickle_path)
            # append_to_dataset_logger(i, cartmass, polemass, polelength, xs.shape, train_logger)

            save_pickle((xs, control_values, cartmass, polemass, polelength), pickle_path)
            append_to_dataset_logger(i, cartmass, polemass, polelength, xs.shape, train_logger)
            
        

        elif i >= args.training.train_steps and i < args.training.train_steps + args.training.test_pendulums:
            cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(size=bsize, cartmass=2, polemasslowerbound=0.3, polemassupperbound=1.0, polelengthlowerbound=1.0, polelengthupperbound=1.5) 
            sampler = CartPoleSampler2(n_dims=4)
            T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, bsize=bsize, cartmass=cartmass, polemass=polemass, polelength=polelength) 
            
            
            pickle_file = f'batch_test_{i-args.training.train_steps}.pkl'
            pickle_path = os.path.join(test_data_dir, pickle_file)

            save_pickle((xs, control_values, cartmass, polemass, polelength), pickle_path)
            append_to_dataset_logger(i-args.training.train_steps, cartmass, polemass, polelength, xs.shape, test_logger)

        else:
            # sampler = CartPoleSampler(n_dims=4)
            # cartmass, polemass, polelength = get_valid_masses_and_lengths_uniform(cartmasslowerbound=2, cartmassupperbound=3, polemasslowerbound=1.6, polemassupperbound=2.6, polelengthlowerbound=1.6, polelengthupperbound=2.6) ### 4/9/2025 (ebonye) cartpole system
            # T, xs, control_values = sampler.generate_xs_dataset(curriculum.n_points, cartmass=cartmass, polemass=polemass, polelength=polelength) ### 4/9/2025 (ebonye) cartpole system
            # pickle_file = f'multipendulum_test_outofdistr_{i-args.training.train_steps-args.training.test_pendulums}.pkl'
            # pickle_path = os.path.join(test_data_dir_outofdistr, pickle_file)
            # save_pickle((xs, control_values, cartmass, polemass, polelength), pickle_path)
            # append_to_dataset_logger(i-args.training.train_steps-args.training.test_pendulums, cartmass, polemass, polelength, xs.shape, test_logger_outofdistr)
            pass
            

        # append_to_seed_file(seed_file, i, seed[0] + i, masses, lengths, k_values, xs.shape)
        curriculum.update()
    
    

def main(args):
    # reseed_all(seed[0])
    global_seed = 42
    random.seed(global_seed)
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

