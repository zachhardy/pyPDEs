import os
import sys
import time
import warnings
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.stats.sampling import NumericalInversePolynomial
from sklearn.model_selection import LeaveOneOut

from typing import Union

from utils import get_reader
from utils import get_dataset
from utils import get_hyperparams

from readers import NeutronicsDatasetReader

from pyROMs.pod import POD_MCI


def exercise_rom(
        X: np.ndarray,
        Y: np.ndarray,
        pod_mci: POD_MCI,
        qoi_function: callable,
        samples: np.ndarray
) -> np.ndarray:
    """
    Exercise the ROM for

    Parameters
    ----------
    dataset : NeutronicsDatasetReader
    pod_mci : POD_MCI
    qoi_function : callable
        A callable expression to evaluate a QoI.
    variables : str or list[str], default None
        The variables from the dataset to fit the POD-MCI model to.
    n_samples : int, default 20000
        The number of queries to perform.
    """

    ##################################################
    # Initialize the POD-MCI ROM
    ##################################################

    print("Initializing and fitting the POD-MCI model...")

    pod_mci.fit(X.T, Y)

    ##################################################
    # Exercise the POD-MCI ROM
    ##################################################

    print("Exercising the ROM...")

    # Exercise the ROM at the sample points
    t_start = time.time()
    qois = np.zeros(n_samples)
    X_pod = pod_mci.predict(samples).T
    for s in range(n_samples):
        qois[s] = qoi_function(X_pod[s])
    t_end = time.time()

    print()
    print(f"Total query time  :\t{(t_end - t_start):.3g} s")
    print(f"Average query time:\t{(t_end - t_start) / n_samples:.3g} s")
    return qois


def get_qoi_function(problem: str, case: int) -> callable:
    """
    Return the QoI function.

    Parameters
    ----------
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}
    case : int

    Returns
    -------
    callable
    """
    if problem != "LRA":
        def func(x: np.ndarray) -> float:
            x = reader.unstack_simulation_vector(x)[0]
            return np.sum(x[-1])
    else:
        def func(x: np.ndarray) -> float:
            x = reader.unstack_simulation_vector(x)[0]
            return np.sum(x[np.argmax(np.sum(x, axis=1))])
    return func


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

    n_samples = 5000
    case_num = 0
    save = False

    # Parse command line
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "n=" in arg:
                n_samples = int(argval)
            elif "case=" in arg:
                case_num = int(argval)
            elif "save=" in arg:
                save = bool(int(argval))

    ##################################################
    # Get the data
    ##################################################

    print("Getting the data...")

    f = get_qoi_function(problem_name, case_num)

    # Get the dataset and reference QoIs
    reader = get_reader(problem_name, study_num)
    data = get_dataset(reader, problem_name, case_num)
    hyperparams = get_hyperparams(problem_name)

    ##################################################
    # Generate the samples
    ##################################################

    print("Generating the samples...")

    rng = np.random.default_rng()
    samples = np.zeros((n_samples, reader.n_parameters))
    for p in range(reader.n_parameters):
        low, high = reader.parameter_bounds[p]
        samples[:, p] = rng.uniform(low, high, n_samples)

    #################################################
    # Query the ROM
    ##################################################

    rom = POD_MCI(**hyperparams)
    rom_qois = exercise_rom(*data, rom, f, samples)
    train_qois = [f(x) for x in data[0]]

    # Display the results
    print()
    print(f"Ref. Mean QoI:\t{np.mean(train_qois):.3g}")
    print(f"Mean QoI     :\t{np.mean(rom_qois):.3g}")
    print(f"Median QoI   :\t{np.median(rom_qois):.3g}")
    print(f"STD QoI      :\t{np.std(rom_qois):.3g}")

    plt.figure()
    plt.ylabel("Probability")
    sb.histplot(rom_qois, bins=20, stat='probability', kde=True, ax=plt.gca())
    plt.tight_layout()

    plt.show()
