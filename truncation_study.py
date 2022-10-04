import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from os.path import splitext
from numpy.linalg import norm
from typing import Union

from utils import get_dataset
from utils import get_hyperparams

from readers import NeutronicsDatasetReader

from pyROMs.pod import POD_MCI


def truncation_study(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        variables: Union[str, list[str]] = None,
        split_method: str = "random",
        test_size: float = 0.2,
        interior_only: bool = False,
        seed: int = None,
        filename: str = None
) -> dict:
    """
    Perform a truncation study to examine error as a function of
    truncation parameter.

    Parameters
    ----------
    dataset : NeutronicsDatasetReader
    pod_mci : POD_MCI
    variables : str or list[str], default None
        The variables from the dataset to fit the POD-MCI model to.
    split_method : str, {'random', 'uniform', 'none'}
        The method used to produce the training and test set. Random
        uses scikit-learn train_test_split, uniform takes every other
        snapshot, and none performs a reconstruction error study.
    test_size : float
        The fraction of samples to use for validation.
    interior_only : bool, default False
        A flag for excluding boundary samples from the test set.
    seed : int or None, default None
        The random number seed.
    filename : str, default None.
        A location to save the plot to, if specified.

    Returns
    -------
    dict
    """

    interp_method = pod_mci.interpolation_method
    if not interior_only:
        if "rbf" not in interp_method and interp_method != "neighbor":
            err = "Only RBF and nearest neighbor interpolants are " \
                  "permissible when extrapolation may be performed."
            raise ValueError(err)

    ##################################################
    # Define the train/test split
    ##################################################

    print("Defining the training and test set...")
    if split_method == "random":
        splits = dataset.train_test_split(
            variables=variables,
            test_size=test_size,
            interior_only=interior_only,
            seed=seed,
        )
        X_train, X_test, Y_train, Y_test = splits

    else:
        Y = dataset.parameters
        X = dataset.create_2d_matrix(variables)

        if split_method == "uniform":
            X_train, X_test = X[::2], X[1::2]
            Y_train, Y_test = Y[::2], Y[1::2]

        elif split_method == "none":
            X_train, X_test = X, X
            Y_train, Y_test = Y, Y

        else:
            raise ValueError(f"{split_method} is not a valid method.")

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

    if split_method != "none":
        plt.semilogy(out["n_modes"], out["mean"], '-*b', label="Mean")
        plt.semilogy(out["n_modes"], out["max"], '-or', label="Max")
        plt.semilogy(out["n_modes"], out["min"], '-+k', label="Min")
    else:
        plt.semilogy(pod_mci.singular_values, '-+k', label="Singular Values")
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
    splitting_method = "random"
    args = [0.2, False, None]
    save = False

    variable_names = "power_density"
    if problem_name == "Sphere3g":
        variable_names = None

    # Parse the command line
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "split=" in arg:
                splitting_method = argval
            elif "test_size=" in arg:
                args[0] = float(argval)
            elif "interior=" in arg:
                args[1] = bool(int(argval))
            elif "seed=" in arg:
                args[2] = int(argval)
            elif "save=" in arg:
                save = bool(int(argval))

    # Get the dataset
    data = get_dataset(problem_name, study_num)

    # Initialize the ROM
    hyperparams = get_hyperparams(problem_name)
    rom = POD_MCI(**hyperparams)

    # Perform the truncation study
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

    truncation_study(data, rom, variable_names,
                     splitting_method, *args, filename=fname)

    plt.show()
