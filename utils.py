import os
import sys
import pickle

from readers import NeutronicsDatasetReader
from readers import NeutronicsSimulationReader


def get_dataset(problem: str, study: int) -> NeutronicsDatasetReader:
    """
    Get the data from the specified parameter study.

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
    path = f"{path}/Problems/{problem}/pickles"
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory.")

    filepath = f"{path}/{study_name}.obj"
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data


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
    if problem == "LRA":
        hyperparams["svd_rank"] = 1.0 - 1.0e-10
    return hyperparams
