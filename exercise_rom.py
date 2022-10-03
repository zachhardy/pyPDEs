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

from utils import get_dataset
from utils import get_reference
from utils import get_hyperparams

from readers import NeutronicsDatasetReader

from pyROMs.pod import POD_MCI


def exercise_rom(
        dataset: NeutronicsDatasetReader,
        pod_mci: POD_MCI,
        qoi_function: callable,
        variables: Union[str, list[str]] = None,
        n_samples: int = int(2.0e4)
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

    Y = dataset.parameters
    X = dataset.create_2d_matrix(variables)
    pod_mci.fit(X.T, Y)

    ##################################################
    # Exercise the POD-MCI ROM
    ##################################################

    print("Setting up samples...")

    # Generate the samples
    rng = np.random.default_rng()
    samples = np.zeros((n_samples, pod_mci.n_parameters))
    for p in range(pod_mci.n_parameters):
        low, high = dataset.parameter_bounds[p]
        samples[:, p] = rng.uniform(low, high, n_samples)

    print("Exercising the ROM...")

    # Exercise the ROM at the sample points
    t_start = time.time()
    qois = np.zeros(n_samples)
    X_pod = pod_mci.predict(samples).T
    X_pod = dataset.unstack_simulation_vector(X_pod)
    for s in range(n_samples):
        qois[s] = qoi_function(X_pod[s])
    t_end = time.time()

    print()
    print(f"Total query time  :\t{(t_end - t_start):.3g} s")
    print(f"Average query time:\t{(t_end - t_start) / n_samples:.3g} s")
    return qois


def get_qoi_function(problem: str) -> callable:
    """
    Return the QoI function.

    Parameters
    ----------
    problem : {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}

    Returns
    -------
    callable
    """
    if problem != "LRA":
        def func(x: np.ndarray) -> float:
            assert x.ndim == 2
            return np.sum(x[-1])
    else:
        def func(x: np.ndarray) -> float:
            assert x.ndim == 2
            return np.sum(x[np.argmax(np.sum(x, axis=1))])
    return func


if __name__ == "__main__":

    if len(sys.argv) < 3:
        msg = "Invalid command line arguments. "
        msg += "A problem name and study number must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    study_num = int(sys.argv[2])
    variable_names = "power_density"
    args = [5000]

    # Parse command line
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "n=" in arg:
                args[0] = int(argval)

    # Define the QoI function
    f = get_qoi_function(problem_name)

    # Get the reference problem
    reference = get_reference(problem_name)
    X_ref = reference.create_simulation_matrix(variable_names)
    ref_qoi = f(X_ref)

    # Get the dataset
    data = get_dataset(problem_name, study_num)

    # Initialize the ROM
    hyperparams = get_hyperparams(problem_name)
    rom = POD_MCI(**hyperparams)

    # Query the ROM
    rom_qois = exercise_rom(data, rom, f, variable_names, *args)

    # Display the results
    print()
    print(f"Reference QoI:\t{ref_qoi:.3g}")
    print(f"Mean QoI     :\t{np.mean(rom_qois):.3g}")
    print(f"STD QoI      :\t{np.std(rom_qois):.3g}")

    plt.figure()
    plt.ylabel("Probability")
    sb.histplot(rom_qois, bins=20, stat='probability', kde=True, ax=plt.gca())
    plt.tight_layout()

    plt.show()
