import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

from typing import Union

from utils import get_dataset
from utils import get_default_params

from readers import NeutronicsDatasetReader

from pyROM.pod import POD_MCI


def truncation_study(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        variables: Union[str, list[str]] = None,
        test_size: float = 0.2,
        interior_only: bool = False,
        seed: int = None
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

    # Output structure
    out = {"tau": [], "n_modes": [], "mean": [],
           "max": [], "min": []}

    print("Starting truncation study...")
    taus = [10.0 ** i for i in range(-16, 0)]
    for tau in taus:

        # Refit the POD-MCI model
        pod_mci.refit(1.0 - tau)
        X_pod = pod_mci.predict(Y_test).T

        errors = np.zeros(len(X_test))
        for i, (x_pod, x_test) in enumerate(zip(X_pod, X_test)):
            errors[i] = norm(x_pod - x_test) / norm(x_test)

        out["tau"].append(tau)
        out["n_modes"].append(pod_mci.n_modes)
        out["mean"].append(np.mean(errors))
        out["max"].append(np.max(errors))
        out["min"].append(np.min(errors))

    ##################################################
    # Plot the results
    ##################################################

    fig: plt.Figure = plt.figure()

    # Plot as a function of number of modes
    ax: plt.Axes = fig.add_subplot(1, 2, 1)
    ax.set_xlabel("n")
    ax.set_ylabel("Relative $\ell_2$ Error")
    ax.semilogy(out["n_modes"], out["mean"], '-*b', label="Mean")
    ax.semilogy(out["n_modes"], out["max"], '-or', label="Max")
    ax.semilogy(out["n_modes"], out["min"], '-+k', label="Min")
    ax.legend()
    ax.grid(True)

    ax: plt.Axes = fig.add_subplot(1, 2, 2)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Relative $\ell_2$ Error")
    ax.loglog(out["tau"], out["mean"], '-*b', label="Mean")
    ax.loglog(out["tau"], out["max"], '-or', label="Max")
    ax.loglog(out["tau"], out["min"], '-+k', label="Min")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return out


if __name__ == "__main__":

    if len(sys.argv) < 3:
        msg = "Invalid command line arguments. "
        msg += "A problem name and study number must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    study_num = int(sys.argv[2])

    defaults = get_default_params(problem_name)
    svd_rank = 1.0 - defaults.pop("tau")
    interpolant = defaults.pop("interpolant")
    variable_names = defaults.pop("variable_names")
    hyperparams = {"epsilon": defaults.pop("epsilon")}

    # Get the dataset
    data = get_dataset(problem_name, study_num)

    # Initialize the ROM
    rom = POD_MCI(svd_rank, interpolant, **hyperparams)

    # Perform the truncation study
    args = [0.2, False, None]
    for arg in sys.argv[1:]:

        if "test_size=" in arg:
            args[0] = float(arg.split("=")[1])
            if args[0] <= 0.0 or args[0] >= 1.0:
                msg = "Test fraction must be between 0 and 1."
                raise ValueError(msg)

        if "interior=" in arg:
            args[1] = int(arg.split("=")[1])
            if args[1] not in [0, 1]:
                msg = "interior argument must be 0 or 1."
                raise ValueError(msg)
            args[1] = bool(args[1])

        if "seed=" in arg:
            args[2] = arg.split("=")[1]
            try:
                args[2] = int(args[2])
            except TypeError:
                args[2] = None

    truncation_study(data, rom, variable_names, *args)

    plt.show()
