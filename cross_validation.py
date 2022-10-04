import os
import sys
import time
import itertools

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from numpy.linalg import norm
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut

from os.path import splitext
from typing import Union

from utils import get_dataset
from utils import get_hyperparams

from readers import NeutronicsDatasetReader
from readers import NeutronicsSimulationReader

from pyROMs.pod import POD_MCI


def cross_validation(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        variables: Union[str, list[str]] = None,
        cv_method: str = "loo",
        interior_only: bool = False,
        n_splits: int = 5,
        n_repeats: int = 100,
        seed: int = None,
        filename: str = None
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
    n_splits : int, default 5
        The number of splits for k-fold cross-validation.
    n_repeats : int, default 100
        The number of repeats for k-fold cross-validation
    seed : int, default None
        The random number seed.
    filename : str, default None.
        A location to save the plot to, if specified.

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
    out = {"mean": [], "max": [], "min": [], "all": [],
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
            out["all"].append(errs[i])

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

        fig: plt.Figure = plt.figure()

        # Plot mean errors
        ax: plt.Axes = fig.add_subplot(1, 2, 1)
        ax.set_xlabel("Mean Errors")
        ax.set_ylabel("Probability")
        sb.histplot(out["mean"], bins=10, stat="probability",
                    log_scale=True, ax=ax)

        # Plot max errors
        ax: plt.Axes = fig.add_subplot(1, 2, 2)
        ax.set_xlabel("Max Error")
        ax.set_ylabel("Probability")
        sb.histplot(out["max"], bins=10, stat="probability",
                    log_scale=True, ax=ax)

        fig.tight_layout()

        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(f"{base}.pdf")

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
    args = ["loo", False, 5, 20, None]
    save = False

    variable_names = "power_density"
    if problem_name == "Sphere3g":
        variable_names = None

    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "cv=" in arg:
                args[0] = argval
            elif "interior=" in arg:
                args[1] = bool(int(argval))
            elif "nsplits=" in arg:
                args[2] = int(argval)
            elif "nrepeats=" in arg:
                args[3] = int(argval)
            elif "seed=" in arg:
                args[4] = int(argval)
            elif "save=" in arg:
                save = bool(int(argval))

    # Get the dataset
    data = get_dataset(problem_name, study_num)

    # Initialize the ROM
    hyperparams = get_hyperparams(problem_name)
    rom = POD_MCI(**hyperparams)

    # Perform cross-validation
    fname = None
    if save:
        fname = f"/Users/zhardy/Documents/Journal Papers/POD-MCI/figures/"

        if problem_name == "Sphere3g":
            fname += f"{problem_name}/rom/"
            fname += "oned/" if study_num == 0 else "threed/"
            fname += "error_distribution.pdf"

    cross_validation(data, rom, variable_names, *args, filename=fname)

    plt.show()
