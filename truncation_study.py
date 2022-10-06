import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from os.path import splitext
from numpy.linalg import norm
from typing import Union

from sklearn.model_selection import train_test_split

from utils import get_reader
from utils import get_dataset
from utils import get_hyperparams
from utils import train_test_split

from readers import NeutronicsDatasetReader

from pyROMs.pod import POD_MCI


def truncation_study(
        X_train: np.ndarray,
        X_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
        pod_mci: POD_MCI,
        filename: str = None
) -> dict:
    """
    Perform a truncation study to examine error as a function of
    truncation parameter.

    Parameters
    ----------
    X_train : numpy.ndarray
    X_test : numpy.ndarray
    Y_train : numpy.ndarray
    Y_test : numpy.ndarray
    pod_mci : POD_MCI
    filename : str, default None.
        A location to save the plot to, if specified.

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

    vals = [i + 1 for i in range(pod_mci.n_snapshots)]

    # Output structure
    out = {"tau": [], "n_modes": [], "mean": [],
           "max": [], "min": []}

    print("Starting truncation study...")
    total_error = np.zeros(len(vals))
    for i, val in enumerate(vals):

        # Refit the POD-MCI model
        pod_mci.refit(val)
        X_pod = pod_mci.predict(Y_test).T

        total_error[i] = norm(X_pod - X_test) / norm(X_test)
        errors = np.zeros(len(X_test))
        for n, (x_pod, x_test) in enumerate(zip(X_pod, X_test)):
            errors[n] = norm(x_pod - x_test) / norm(x_test)

        svals = pod_mci.singular_values
        tau = sum(svals[val + 1:] ** 2) / sum(svals ** 2)

        out["tau"].append(tau)
        out["n_modes"].append(pod_mci.n_modes)
        out["mean"].append(np.mean(errors))
        out["max"].append(np.max(errors))
        out["min"].append(np.min(errors))

    ##################################################
    # Plot the results
    ##################################################

    plt.figure()
    plt.xlabel("n")
    plt.ylabel("Relative $\ell_2$ Error")

    if not np.array_equal(X_train, X_test):
        plt.semilogy(out["n_modes"], out["mean"], '-*b', label="Mean")
        plt.semilogy(out["n_modes"], out["max"], '-or', label="Max")
        plt.semilogy(out["n_modes"], out["min"], '-+k', label="Min")
    else:
        svals = pod_mci.singular_values
        plt.semilogy(svals/sum(svals), '-+k', label="Singular Values")
        plt.semilogy(total_error, '-*b', label="Reconstruction Error")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        base, ext = splitext(filename)
        plt.savefig(f"{base}.pdf")

    return out


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
    interior = False
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
        splits = (X, X, Y, Y)

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

    # Define output filename
    fname = None
    if save:
        fname = f"/Users/zhardy/Documents/Journal Papers/POD-MCI/figures/"

        if problem_name == "Sphere3g":
            fname += f"{problem_name}/rom/"
            fname += "oned/" if study_num == 0 else "threed/"

            if splitting_method == "none":
                fname += "verification.pdf"
            else:
                fname += f"truncation_{splitting_method}.pdf"

    rom = POD_MCI(**hyperparams)
    truncation_study(*splits, rom, filename=fname)

    plt.show()
