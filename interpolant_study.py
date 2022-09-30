import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

from typing import Union

from utils import get_dataset
from utils import get_hyperparams

from readers import NeutronicsDatasetReader

from pyROMs.pod import POD_MCI


def interpolant_study(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        variables: Union[str, list[str]] = None,
        test_size: float = 0.2,
        interior_only: bool = False,
        seed: int = None
) -> dict:
    """
    Perform an interpolant study to examine error as a function of
    interpolation method.

    Parameters
    ----------
    dataset : NeutronicsDatasetReader
    pod_mci : POD_MCI
    variables : str or list[str], default None
        The variables from the dataset to fit the POD-MCI model to.
    test_size : float
        The fraction of samples to use for validation.
    interior_only : bool, default False
        A flag for excluding boundary samples from the test set.
    seed : int or None, default None
        The random number seed.

    Returns
    -------
    dict
    """

    ##################################################
    # Define the train/test split
    ##################################################

    print("Defining the training and test set...")
    splits = dataset.train_test_split(
        variables=variables,
        test_size=test_size,
        interior_only=interior_only,
        seed=seed,
    )

    X_train, X_test, Y_train, Y_test = splits

    ##################################################
    # Fitting the POD-MCI model
    ##################################################

    print("Fitting the POD-MCI model...")
    pod_mci.fit(X_train.T, Y_train)

    ##################################################
    # Run the study
    ##################################################

    interpolants = [
        "rbf_linear", "rbf_cubic", "rbf_quintic",
        "rbf_thin_plate_spline", "nearest"
    ]

    # Output structure
    out = {"method": interpolants,
           "mean": [], "max": [], "min": []}

    print("Starting interpolant study...")
    for method in interpolants:

        # Refit the POD-MCI model
        pod_mci.refit(pod_mci.svd_rank, method)
        X_pod = pod_mci.predict(Y_test).T

        errors = np.zeros(len(X_test))
        for i, (x_pod, x_test) in enumerate(zip(X_pod, X_test)):
            errors[i] = norm(x_pod - x_test) / norm(x_test)

        out["mean"].append(np.mean(errors))
        out["max"].append(np.max(errors))
        out["min"].append(np.min(errors))

    print()
    print(f"-" * 85)
    print(f"{'Interpolant':<25}{'Mean Error':<20}"
          f"{'Max Error':<20}{'Min Error':<20}")
    print(f"-"*85)

    vals = (out["method"], out["mean"], out["max"], out["min"])
    for (method, emean, emax, emin) in zip(*vals):
        method = method.split("_")
        if method[0] == "rbf":
            for i in range(len(method[1:])):
                method[i + 1] = method[i + 1].capitalize()
            method = " ".join(method[1:]) + " RBF"
        elif method[0] == "nearest":
            method = "Nearest Neighbor"

        print(f"{method:<25}{emean:<20.3g}{emax:<20.3g}{emin:<20.3g}")
    print()


if __name__ == "__main__":

    if len(sys.argv) < 3:
        msg = "Invalid command line arguments. "
        msg += "A problem name and study number must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    study_num = int(sys.argv[2])
    variable_names = "power_density"
    args = [0.2, False, None]

    # Parse the command line
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "test_size=" in arg:
                args[0] = float(argval)
            elif "interior=" in arg:
                args[1] = bool(int(argval))
            elif "seed=" in arg:
                args[2] = int(argval)

    # Get the dataset
    data = get_dataset(problem_name, study_num)

    # Initialize the ROM
    hyperparams = get_hyperparams(problem_name)
    rom = POD_MCI(**hyperparams)

    # Perform the interpolant study
    interpolant_study(data, rom, variable_names, *args)
