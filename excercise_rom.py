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

from utils import get_data, get_default_setup
from readers import NeutronicsDatasetReader
from pyROM.pod import POD_MCI


def exercise_rom(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        qoi_function: callable,
        variables: Union[str, list[str]] = None,
        n: int = int(2.0e4)
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
    n : int, default 20000
        The number of queries to perform.
    """

    ##################################################
    # Initialize the POD-MCI ROM
    ##################################################

    print("Initializing and fitting the POD-MCI model...")

    Y = dataset.parameters
    X = dataset.create_2d_matrix(variables)
    pod_mci.fit(X.T, Y)

    ##################################################
    # Exercise the POD-MCI ROM
    ##################################################

    print("Setting up samples...")

    # Generate the samples
    rng = np.random.default_rng()
    samples = np.zeros((n, pod_mci.n_parameters))
    for p in range(pod_mci.n_parameters):
        low, high = dataset.parameter_bounds[p]
        samples[:, p] = rng.uniform(low, high, n)

    print("Exercising the ROM...")

    # Exercise the ROM at the sample points
    t_start = time.time()
    qois = np.zeros(n)
    X_pod = pod_mci.predict(samples).T
    X_pod = dataset.unstack_simulation_vector(X_pod)
    for s in range(n):
        qois[s] = qoi_function(X_pod[s])
    t_end = time.time()

    print(f"Average ROM query took {(t_end - t_start) / n:.3g} s.")
    print(f"Mean QoI:\t{np.mean(qois):.3g}")
    print(f"STD QoI :\t{np.std(qois):.3g}")

    plt.figure()
    plt.ylabel("Probability")
    sb.histplot(qois, bins=50, stat='probability', kde=True, ax=plt.gca())
    plt.tight_layout()
    return qois


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

    # Defining the QoI functions
    if problem_name in ["Sphere3g", "InfiniteSlab", "TWIGL"]:
        def f(x):
            assert x.ndim == 2
            return np.sum(x[-1])

    elif problem_name == "LRA":
        def f(x):
            assert x.ndim == 2
            return np.sum(x[np.argmax(np.sum(x, axis=1))])

    else:
        msg = f"{problem_name} is not a valid problem name."
        raise AssertionError(msg)

    # Exercise the ROM
    n_samples = 5000
    for arg in sys.argv[1:]:
        if "n=" in arg:
            n_samples = int(arg.split("=")[1])

    exercise_rom(data, rom, f, variable_names, n=n_samples)

    plt.show()
