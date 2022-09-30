import os
import sys
import time
import itertools

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut

from typing import Union

from utils import get_data, get_default_setup
from readers import NeutronicsDatasetReader
from readers import NeutronicsSimulationReader
from pyROM.pod import POD_MCI


def cross_validation(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        variables: Union[str, list[str]] = None,
        cv_method: str = "kfold",
        interior_only: bool = False,
        **kwargs
) -> dict:
    """
   Perform a cross-validation study.

    Parameters
    ----------
    dataset : NeutronicsDatasetReader
    pod_mci : POD_MCI
    variables : str or list[str], default None
        The variables from the dataset to fit the POD-MCI model to.
    cv_method : {'kfold', 'loo'}, default 'kfold'
        The cross-validation method to use. Default used repeated k-fold
        cross-validation. Other option is leave-one-out cross-validation.
    interior_only : bool, default False
        A flag to include only interior samples in the validation set.
    kwargs : varies
        Hyper-parameters for the cross-validator. This includes the number
        of splits, number of repeats, and seed.

    Returns
    -------
    dict
    """

    if cv_method not in ["kfold", "loo"]:
        err = f"{cv_method} is not a valid cross-validation method."
        raise AssertionError(err)

    interp_method = pod_mci.interpolation_method
    if not interior_only:
        if "rbf" not in interp_method and interp_method != "neighbor":
            err = "Only RBF and nearest neighbor interpolants are " \
                  "permissible when extrapolation may be performed."
            raise ValueError(err)

    ##################################################
    # Define the cross-validator
    ##################################################

    print("Setting up the cross-validation sets...")

    # Initialize the cross-validator
    if cv_method == "kfold":
        n_splits, n_repeats, seed = 5, 20, None
        if "n_splits" in kwargs:
            n_splits = kwargs.pop("n_splits")
        if "n_repeats" in kwargs:
            n_repeats = kwargs.pop("n_repeats")
        if "seed" in kwargs:
            seed = kwargs.pop("seed")

        cv = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats,
                           random_state=seed)
    else:
        cv = LeaveOneOut()

    # Define the sets
    Y = dataset.parameters
    X = dataset.create_2d_matrix(variables)
    if interior_only:
        interior = dataset.interior_mask
        iterator = cv.split(X[interior], Y[interior])
    else:
        iterator = cv.split(X, Y)
    iterator, tmp_iterator = itertools.tee(iterator)
    n_cv = sum(1 for _ in tmp_iterator)

    ##################################################
    # Perform cross-validation
    ##################################################

    # Output structure
    out = {"mean": [], "max": [], "min": [],
           "construction_time": [], "query_time": []}

    # Start cross-validating
    print("Starting cross-validation...")
    cv_count, decade = 0, 1
    for train, test in iterator:

        # Define training and test sets
        if interior_only:
            interior, bndry = dataset.interior_mask, dataset.boundary_mask
            X_train, Y_train = X[interior][train], Y[interior][train]
            X_test, Y_test = X[interior][test], Y[interior][test]
            X_train = np.vstack((X_train, X[bndry]))
            Y_train = np.vstack((Y_train, Y[bndry]))
        else:
            X_train, Y_train = X[train], Y[train]
            X_test, Y_test = X[test], Y[test]

        # Construct ROM
        t_start = time.time()
        pod_mci.fit(X_train.T, Y_train)
        t_end = time.time()
        t_construct = t_end - t_start

        # Make predictions
        t_start = time.time()
        X_pod = pod_mci.predict(Y_test).T
        t_end = time.time()
        t_query = t_end - t_start

        # Compute errors
        errs = np.zeros(len(X_pod))
        for i, (x_pod, x_test) in enumerate(zip(X_pod, X_test)):
            errs[i] = norm(x_pod - x_test) / norm(x_test)

        out["mean"].append(np.mean(errs))
        out["max"].append(np.max(errs))
        out["min"].append(np.min(errs))
        out["construction_time"].append(t_construct)
        out["query_time"].append(t_query)

        cv_count += 1
        if cv_count / n_cv >= decade/10.0:
            print(f"Finished {decade*10.0:2g}% of CV sets...")
            decade += 1

    print()
    if cv_method == "kfold":
        print("--- K-Fold Cross-Validation Summary")
    else:
        print("--- Leave-One-Out Cross-Validation Summary")
    print(f"{'# of CV Sets':<30}: {len(out['mean'])}")
    print(f"{'# of POD Modes':<30}: {pod_mci.n_modes}")
    print(f"{'# of Snapshots':<30}: {pod_mci.n_snapshots}")
    print(f"{'Average Construction Time':<30}: "
          f"{np.mean(out['construction_time']):.3g} s")
    print(f"{'Average Query Time':<30}: "
          f"{np.mean(out['query_time']):.3g} s")

    print()
    if cv_method == "kfold":
        print("--- K-Fold Cross-Validation Statistics")

        ci = np.percentile(out["mean"], [2.5, 97.5])
        print("Mean Error Statistics:")
        print(f"\tMean  :\t{np.mean(out['mean']):.3g}")
        print(f"\tMedian:\t{np.median(out['mean']):.3g}")
        print(f"\t95% CI:\t[{ci[0]:.3g}, {ci[1]:.3g}]")

        ci = np.percentile(out["max"], [2.5, 97.5])
        print("\nMax Error Statistics:")
        print(f"\tMean  :\t{np.mean(out['max']):.3g}")
        print(f"\tMedian:\t{np.median(out['max']):.3g}")
        print(f"\t95% CI:\t[{ci[0]:.3g}, {ci[1]:.3g}]")

        ci = np.percentile(out["min"], [2.5, 97.5])
        print("\nMin Error Statistics:")
        print(f"\tMean  :\t{np.mean(out['min']):.3g}")
        print(f"\tMedian:\t{np.median(out['min']):.3g}")
        print(f"\t95% CI:\t[{ci[0]:.3g}, {ci[1]:.3g}]")
    else:
        print("--- Leave-One-Out Cross-Validation Statistics")

        ci = np.percentile(out['mean'], [2.5, 97.5])
        print("Error Statistics:")
        print(f"\tMean  :\t{np.mean(out['mean']):.3g}")
        print(f"\tMedian:\t{np.median(out['mean']):.3g}")
        print(f"\tMax   :\t{np.max(out['mean']):.3g}")
        print(f"\tMin   :\t{np.min(out['min']):.3g}")
        print(f"\t95% CI:\t[{ci[0]:.3g}, {ci[1]:.3g}]")

    return out


if __name__ == "__main__":

    if len(sys.argv) < 3:
        msg = "Invalid command line arguments. "
        msg += "A problem name and study number must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    study_num = int(sys.argv[2])

    defaults = get_default_setup(problem_name)
    svd_rank = 1.0 - defaults.pop("tau")
    interpolant = defaults.pop("interpolant")
    variable_names = defaults.pop("variable_names")
    hyperparams = {"epsilon": defaults.pop("epsilon")}

    # Get the dataset
    data = get_data(problem_name, study_num)

    # Initialize the ROM
    rom = POD_MCI(svd_rank, interpolant, **hyperparams)

    # Perform cross-validation
    args = ["loo", False]
    for arg in sys.argv[1:]:

        if "method=" in arg:
            args[0] = arg.split("=")[1]
            if args[0] not in ["kfold", "loo"]:
                msg = f"{arg[0]} is not a valid cross-validator."
                raise ValueError(msg)

        if "interior=" in arg:
            args[1] = int(arg.split("=")[1])
            if args[1] not in [0, 1]:
                msg = "interior argument must be 0 or 1."
                raise ValueError(msg)
            args[1] = bool(interior_flag)

    cross_validation(data, rom, variable_names, *args)
