import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

from typing import Union

from utils import get_reader
from utils import get_dataset
from utils import get_hyperparams
from utils import train_test_split

from readers import NeutronicsDatasetReader

from pyROMs.pod import POD_MCI


def interpolant_study(
        X_train: np.ndarray,
        X_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
        pod_mci: POD_MCI,
) -> dict:
    """
    Perform an interpolant study to examine error as a function of
    interpolation method.

    Parameters
    ----------
    X_train : numpy.ndarray
    X_test : numpy.ndarray
    Y_train : numpy.ndarray
    Y_test : numpy.ndarray
    pod_mci : POD_MCI

    Returns
    -------
    dict
    """

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

    # Splitting parameters
    splitting_method = "random"
    test_size = 0.2
    interior = False,
    seed = None

    # Which particular problem
    case = 0

    # Save outputs?
    save = False

    # Parse the command line
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "split=" in arg:
                splitting_method = argval
            elif "test_size=" in arg:
                test_size = float(argval)
            elif "interior=" in arg:
                interior = bool(int(argval))
            elif "seed=" in arg:
                seed = int(argval)
            elif "case=" in arg:
                case = int(argval)
            elif "save=" in arg:
                save = bool(int(argval))

    ##################################################
    # Get the data
    ##################################################

    print("Getting the data...")

    reader = get_reader(problem_name, study_num)
    X, Y = get_dataset(reader, problem_name, case)
    hyperparams = get_hyperparams(problem_name)

    ##################################################
    # Split the data
    ##################################################

    if splitting_method == "random":
        mask = reader.interior_mask if interior else None
        splits = train_test_split(X, Y, test_size, mask, seed)
    elif splitting_method == "uniform":
        splits = (X[::2], X[1::2], Y[::2], Y[1::2])
    else:
        msg = f"{splitting_method} is not a valid splitting method."
        raise ValueError(msg)

    ##################################################
    # Perform truncation study
    ##################################################

    # Last second check
    interp_method = hyperparams["interpolant"]
    if not interior:
        if "rbf" not in interp_method and interp_method != "neighbor":
            msg = "Only RBF and nearest neighbor interpolants are " \
                  "permissible when extrapolation may be performed."
            raise ValueError(msg)

    rom = POD_MCI(**hyperparams)
    interpolant_study(*splits, rom)
