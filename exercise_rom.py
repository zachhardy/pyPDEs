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
from utils import get_default_params

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

    # Get the reference problem
    reference = get_reference(problem_name)
    X_ref = reference.create_simulation_matrix(variable_names)

    # Get the dataset
    data = get_dataset(problem_name, study_num)

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
    n = 5000
    for arg in sys.argv[1:]:
        if "n=" in arg:
            n = int(arg.split("=")[1])

    # Get QoIs
    ref_qoi = f(X_ref)
    rom_qois = exercise_rom(data, rom, f, variable_names, n_samples=n)

    print()
    print(f"Reference QoI:\t{ref_qoi:.3g}")
    print(f"Mean QoI     :\t{np.mean(rom_qois):.3g}")
    print(f"STD QoI      :\t{np.std(rom_qois):.3g}")

    plt.figure()
    plt.ylabel("Probability")
    sb.histplot(rom_qois, bins=50, stat='probability', kde=True, ax=plt.gca(), log_scale=True)
    plt.tight_layout()

    plt.show()
