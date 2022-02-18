"""
Support functions for studies.
"""
__all__ = ['setup_directory', 'setup_range', 'get_data']

import os
import numpy as np
from numpy import ndarray
from readers import NeutronicsDatasetReader
from typing import List, Tuple


def setup_directory(path: str, clear: bool = False):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0 and clear:
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
    n_args = [1, 2, 1, 1]
    if len(args) != n_args[idx]:
        raise ValueError(
            f"Invalid number of arguments for {problem_name}.")

    # Check case argument
    case = int(args[0])
    case_arg_limit = [None, 2, None, None]
    if case_arg_limit[idx] is not None:
        if case > case_arg_limit[idx]:
            raise ValueError(
                f"Invalid case argument for {problem_name}.")

    # Check study argument
    study_arg_limit = [3, 6, 6, 6]
    study = int(args[1]) if idx == 1 else int(args[0])
    if study > study_arg_limit[idx]:
        raise ValueError(
            f"Invalid study argument for {problem_name}.")

    # Get path to here
    path = os.path.dirname(os.path.abspath(__file__))

    # Handle three group sphere
    if problem_name == 'three_group_sphere':
        if study == 0:
            study_name = 'size'
        elif study == 1:
            study_name = 'density'
        elif study == 2:
            study_name = 'size_density'
        else:
            study_name = 'size_density_down_scatter'

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

    # Handle TWIGL
    if problem_name == 'twigl':
        if study == 0:
            study_name = 'magnitude'
        elif study == 1:
            study_name = 'duration'
        elif study == 2:
            study_name = 'scatter'
        elif study == 3:
            study_name = 'magnitude_duration'
        elif study == 4:
            study_name = 'magnitude_scatter'
        elif study == 5:
            study_name = 'duration_scatter'
        else:
            study_name = 'magnitude_duration_scatter'

    # Hande LRA
    if problem_name == 'lra':
        if study == 0:
            study_name = 'magnitude'
        elif study == 1:
            study_name = 'duration'
        elif study == 2:
            study_name = 'feedback'
        elif study == 3:
            study_name = 'magnitude_duration'
        elif study == 4:
            study_name = 'magnitude_feedback'
        elif study == 5:
            study_name = 'duration_feedback'
        else:
            study_name = 'magnitude_duration_feedback'

    # Define the path to the case and study
    path = f"{path}/{problem_name}/outputs/"
    if case_arg_limit[idx] is not None:
        path += f"{case_name}/"
    path += f"{study_name}"
    if not os.path.isdir(path):
        raise NotADirectoryError('Invalid path.')

    # Create the data set
    dataset = NeutronicsDatasetReader(path)
    dataset.read_dataset()
    return dataset
