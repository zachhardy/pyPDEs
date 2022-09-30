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
    if problem not in ["Sphere3g", "InfiniteSlab", "TWIGL", "LRA"]:
        raise ValueError(f"{problem} is not a valid problem.")

    study = int(study)
    if study > 6:
        raise ValueError(f"{study} is an invalid study number.")

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
        else:
            study_name = "radius_density_scatter"

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
        else:
            study_name = "magnitude_duration_interface"

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
        else:
            study_name = "magnitude_duration_scatter"

    else:
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
        else:
            study_name = "magnitude_duration_feedback"

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


def get_default_params(problem: str) -> dict:
    """
    Return the default hyper-parameters for each problem.

    Parameters
    ----------
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}

    Returns
    -------
    dict
    """
    params = {"tau": 1.0e-8,
              "interpolant": "rbf",
              "variable_names": "power_density"}
    if problem == "Sphere3g":
        params["variable_names"] = None
        params["epsilon"] = 200.0
    elif problem == "InfiniteSlab":
        params["epsilon"] = 10.0
    elif problem == "TWIGL":
        params["epsilon"] = 20.0
    elif problem == "LRA":
        params["tau"] = 1.0e-10
        params["epsilon"] = 200.0
    else:
        err = f"{problem} is an invalid problem."
        raise ValueError(err)
    return params
