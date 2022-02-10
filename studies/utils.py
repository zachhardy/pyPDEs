"""
Support functions for studies.
"""
__all__ = ['setup_directory', 'setup_range', 'get_data']

import os
import numpy as np
from numpy import ndarray
from readers import NeutronicsDatasetReader
from typing import List, Tuple


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f'rm -r {path}/*')


def setup_range(ref: float, var: float, N: int):
    return ref * (1.0 + var*np.linspace(-1.0, 1.0, N))


def get_data(problem_name: str, *args) -> NeutronicsDatasetReader:
    # Is this a valid problem name?
    options = ['three_group_sphere', 'infinite_slab', 'twigl', 'lra']
    if problem_name not in options:
        raise ValueError("Invalid problem name.")
    idx = options.index(problem_name)

    # Have the correct number of arguments been supplied?
    n_args = [2, 2, 2, 1]
    if len(args) != n_args[idx]:
        raise ValueError(
            f"Invalid number of arguments for {problem_name}.")

    # Check case argument
    case = int(args[0])
    case_arg_limit = [1, 2, 1, None]
    if case_arg_limit[idx] is not None:
        if case > case_arg_limit[idx]:
            raise ValueError(
                f"Invalid case argument for {problem_name}.")

    # Check study argument
    study_arg_limit = [3, 6, 2, 2]
    study = int(args[1]) if idx < 3 else int(args[0])
    if study > study_arg_limit[idx]:
        raise ValueError(
            f"Invalid study argument for {problem_name}.")

    # Get path to here
    path = os.path.dirname(os.path.abspath(__file__))

    # Handle three group sphere
    if problem_name == 'three_group_sphere':
        case_name = 'keigenvalue' if case == 0 else 'ics'

        if study == 0:
            study_name = 'density'
        elif study == 1:
            study_name = 'size'
        elif study == 2:
            study_name = 'density_size'
        else:
            study_name = 'density_size_down_scatter'

    # Handle infinite slab
    if problem_name == 'infinite_slab':
        if args[0] == 0:
            case_name = 'subcritical'
        elif args[0] == 1:
            case_name = 'supercritical'
        else:
            case_name = 'prompt_supercritical'

        if study == 0:
            study_name = 'magnitude'
        elif study == 1:
            study_name = 'duration'
        elif study == 2:
            study_name = 'interface'
        elif study == 3:
            study_name = 'magnitude_duration'
        elif study == 4:
            study_name = 'magnitude_interface'
        elif study == 5:
            study_name = 'duration_interface'
        else:
            study_name = 'magnitude_duration_interface'

    # Define the path to the case and study
    path = f"{path}/{problem_name}/outputs/{case_name}/{study_name}"
    if not os.path.isdir(path):
        raise NotADirectoryError('Invalid path.')

    # Create the data set
    dataset = NeutronicsDatasetReader(path)
    dataset.read_dataset()
    return dataset
