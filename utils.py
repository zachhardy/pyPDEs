import os
import sys
import pickle

import numpy as np

from readers import NeutronicsDatasetReader
from readers import NeutronicsSimulationReader


def get_reader(
        problem: str,
        study: int,
        validation: bool = False
) -> NeutronicsDatasetReader:
    """
    Get the reader from the specified parameter study.

    Parameters
    ----------
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}
    study : int
        The index of the parameter study.

    Returns
    -------
    NeutronicsDatasetReader
    """

    ##################################################
    # Define the name of the study
    ##################################################

    if problem == "Sphere3g":
        if study == 0:
            study_name = "radius"
        elif study == 1:
            study_name = "density"
        elif study == 2:
            study_name = "scatter"
        elif study == 3:
            study_name = "radius_density"
        elif study == 4:
            study_name = "radius_scatter"
        elif study == 5:
            study_name = "density_scatter"
        elif study == 6:
            study_name = "radius_density_scatter"
        else:
            raise ValueError(f"{study} is an invalid study number.")

    elif problem == "InfiniteSlab":
        if study == 0:
            study_name = "magnitude"
        elif study == 1:
            study_name = "duration"
        elif study == 2:
            study_name = "interface"
        elif study == 3:
            study_name = "magnitude_duration"
        elif study == 4:
            study_name = "magnitude_interface"
        elif study == 5:
            study_name = "duration_interface"
        elif study == 6:
            study_name = "magnitude_duration_interface"
        else:
            raise ValueError(f"{study} is an invalid study number.")

    elif problem == "TWIGL":
        if study == 0:
            study_name = "magnitude"
        elif study == 1:
            study_name = "duration"
        elif study == 2:
            study_name = "scatter"
        elif study == 3:
            study_name = "magnitude_duration"
        elif study == 4:
            study_name = "magnitude_scatter"
        elif study == 5:
            study_name = "duration_scatter"
        elif study == 6:
            study_name = "magnitude_duration_scatter"
        else:
            raise ValueError(f"{study} is an invalid study number.")

    elif problem == "LRA":
        if study == 0:
            study_name = "magnitude"
        elif study == 1:
            study_name = "duration"
        elif study == 2:
            study_name = "feedback"
        elif study == 3:
            study_name = "magnitude_duration"
        elif study == 4:
            study_name = "magnitude_feedback"
        elif study == 5:
            study_name = "duration_feedback"
        elif study == 6:
            study_name = "magnitude_duration_feedback"
        else:
            raise ValueError(f"{study} is an invalid study number.")

    else:
        raise ValueError(f"{problem} is not a valid problem.")

    ##################################################
    # Unpickle the data handler
    ##################################################

    path = os.path.abspath(os.path.dirname(__file__))
    path = f"{path}/Problems/{problem}/pickles/"
    path += "training" if not validation else "validation"
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory.")

    filepath = f"{path}/{study_name}.obj"
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


def get_dataset(
        reader: NeutronicsDatasetReader,
        problem: str,
        case: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the appropriate data for a POD-MCI ROM.

    Parameters
    ----------
    reader : NeutronicsDatasetReader
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}
    case : int
        The case for the POD-MCI ROM.

    Returns
    -------
    X : numpy.ndarray
        The simulation data.
    Y : numpy.ndarray
        The parameters corresponding to each simulation.
    """
    if case == 0:
        if problem == "Sphere3g":
            X = reader.create_2d_matrix(None)
        else:
            X = reader.create_2d_matrix("power_density")

    elif problem == "Sphere3g":
        if case == 1:
            X = reader.create_3d_matrix("power_density")[:, -1]
        elif case == 2:
            X = np.array([sim.powers for sim in reader])
        else:
            raise NotImplementedError

    elif problem == "LRA":
        if case == 1:
            X = reader.create_3d_matrix("power_density")
            X = np.array([x[np.argmax(np.sum(x, axis=1))] for x in X])
        elif case == 2:
            X = np.array([sim.powers for sim in reader])
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    return X, reader.parameters


def get_reference(problem: str) -> NeutronicsSimulationReader:
    """
    Return the reference solution for the specified problem.

    Parameters
    ----------
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}

    Returns
    -------
    NeutronicsSimulationReader
    """
    if problem not in ["Sphere3g", "InfiniteSlab", "TWIGL", "LRA"]:
        raise ValueError(f"{problem} is not a valid problem.")

    path = os.path.abspath(os.path.dirname(__file__))
    path = f"{path}/Problems/{problem}/outputs"
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory.")
    return NeutronicsSimulationReader(path).read()


def get_hyperparams(problem: str) -> dict:
    """
    Return the default hyper-parameters for each problem.

    Parameters
    ----------
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}

    Returns
    -------
    dict
    """
    hyperparams = {"svd_rank": 1.0 - 1.0e-8,
                   "interpolant": "rbf",
                   "neighbors": None,
                   "epsilon": 100.0}
    return hyperparams


def train_test_split(
        X: np.ndarray,
        Y: np.ndarray,
        test_size: float = 0.2,
        interior_mask: list[bool] = None,
        seed: int = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return training and test sets.

    Parameters
    ----------
    X : numpy.ndarray
        The simulation data
    Y : numpy.ndarray
        The parameters corresponding to the simulation data.
    test_size : float, default 0.2
        The fraction of simulations to reserve for validation.
    interior_mask : list[bool], default None
        A mask for interior simulations. If not None, the validation
        set will only contain interior simulations.
    seed : int, default None
        Random number seed.

    Returns
    -------
    numpy.ndarray
        The training simulation data.
    numpy.ndarray
        The validation simulation data.
    numpy.ndarray
        The training simulation parameters.
    numpy.ndarray
        The validation simulation parameters.
    """
    from sklearn.model_selection import train_test_split

    if interior_mask is not None:
        splits = train_test_split(
            X[interior_mask], Y[interior_mask],
            test_size=test_size, random_state=seed
        )

        bndry_mask = [not flag for flag in interior_mask]
        splits[0] = np.vstack((splits[0], X[bndry_mask]))
        splits[2] = np.vstack((splits[2], Y[bndry_mask]))

    else:
        splits = train_test_split(
            X, Y, test_size=test_size, random_state=seed
        )

    return splits
