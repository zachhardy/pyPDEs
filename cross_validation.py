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

from utils import get_reader
from utils import get_dataset
from utils import get_hyperparams

from readers import NeutronicsDatasetReader
from readers import NeutronicsSimulationReader

from pyROMs.pod import POD_MCI


def cross_validation(
        X: np.ndarray,
        Y: np.ndarray,
        cross_validator: Union[RepeatedKFold, LeaveOneOut],
        pod_mci: POD_MCI,
        interior_mask: list[bool] = None,
        logscale: bool = False,
        filename: str = None
) -> dict:
    """
   Perform a cross-validation study.

    Parameters
    ----------
    X : numpy.ndarray
        The dataset to model using POD-MCI.
    Y : numpy.ndarray
        The parameters corresponding to the snapshots of `X`.
    cross_validator : RepeatedKFold or LeaveOneOut
        A cross validator object to use.
    pod_mci : POD_MCI
        The POD-MCI ROM initialized with the desired parameters.
    interior_mask : list[bool], default None
        A mask to extract interior snapshots. If not None, this
        causes only interior snapshots to be in validation sets.
    logscale : bool, default False
        A flag for a log scale histogram.
    filename : str, default None.
        A location to save the plot to, if specified.

    Returns
    -------
    dict
    """

    interp_method = hyperparams["interpolant"]
    if interior_mask is not None:
        if "rbf" not in interp_method and interp_method != "neighbor":
            err = "Only RBF and nearest neighbor interpolants are " \
                  "permissible when extrapolation may be performed."
            raise ValueError(err)

    ##################################################
    # Define the cross-validation sets
    ##################################################

    if interior_mask is not None:
        iterator = cross_validator.split(X[interior_mask])
    else:
        iterator = cross_validator.split(X)
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
        if (interior_mask is not None and
                isinstance(cross_validator, RepeatedKFold)):
            bndry = [not flag for flag in interior_mask]
            X_train = np.vstack((X[interior_mask][train], X[bndry]))
            Y_train = np.vstack((Y[interior_mask][train], Y[bndry]))

            X_test = X[interior_mask][test]
            Y_test = Y[interior_mask][test]
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
        print(f"\tMax   :\t{np.max(out['mean']):.3g}")
        print(f"\tMedian:\t{np.median(out['mean']):.3g}")
        print(f"\t95% CI:\t[{ci[0]:.3g}, {ci[1]:.3g}]")

        ci = np.percentile(out["max"], [2.5, 97.5])
        print("\nMax Error Statistics:")
        print(f"\tMean  :\t{np.mean(out['max']):.3g}")
        print(f"\tMax   :\t{np.max(out['max']):.3g}")
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
                    log_scale=logscale, ax=ax)

        # Plot max errors
        ax: plt.Axes = fig.add_subplot(1, 2, 2)
        ax.set_xlabel("Max Error")
        ax.set_ylabel("Probability")
        sb.histplot(out["max"], bins=10, stat="probability",
                    log_scale=logscale, ax=ax)

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

    ##################################################
    # Parse inputs
    ##################################################

    if len(sys.argv) < 3:
        msg = "Invalid command line arguments. "
        msg += "A problem name and study number must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    study_num = int(sys.argv[2])

    # Cross-validation parameters
    cv_method = "loo"
    n_splits = 3
    n_repeats = 250
    seed = None
    interior = False
    logscale = True

    # Which particular problem
    case = 0

    # Save outputs?
    save = False

    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "cv=" in arg:
                cv_method = argval
            elif "interior=" in arg:
                interior = bool(int(argval))
            elif "logscale=" in arg:
                logscale = bool(int(argval))
            elif "nsplits=" in arg:
                n_splits = int(argval)
            elif "nrepeats=" in arg:
                n_repeats = int(argval)
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
    data = get_dataset(reader, problem_name, case)
    hyperparams = get_hyperparams(problem_name)

    ##################################################
    # Create the cross-validator
    ##################################################

    # Create the cross-validator
    if cv_method == "kfold":
        cv = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats,
                           random_state=seed)
    elif cv_method == "loo":
        cv = LeaveOneOut()
    else:
        raise NotImplementedError

    ##################################################
    # Perform cross validation
    ##################################################

    # Define output filename
    fname = None
    if save:
        fname = f"/Users/zhardy/Documents/Journal Papers/POD-MCI/figures/"
        if problem_name == "Sphere3g":
            fname += f"{problem_name}/rom/"
            fname += "oned/" if study_num == 0 else "threed/"
            fname += "error_distribution.pdf"

    rom = POD_MCI(**hyperparams)
    mask = None if not interior else reader.interior_mask
    cross_validation(*data, cv, rom, mask, logscale=logscale, filename=fname)

    plt.show()
