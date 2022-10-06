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

    case = 0
    save = False

    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            argval = arg.split("=")[1]
            if "save=" in arg:
                save = bool(int(argval))
            elif "case=" in arg:
                case = int(argval)

    ##################################################
    # Get the data
    ##################################################

    print("Getting the data...")

    training = get_reader(problem_name, study_num, False)
    X_train, Y_train = get_dataset(training, problem_name, case)

    validation = get_reader(problem_name, study_num, True)
    X_test, Y_test = get_dataset(validation, problem_name, case)

    hyperparams = get_hyperparams(problem_name)

    ##################################################
    # Initialize and fit the ROM
    ##################################################

    rom = POD_MCI(**hyperparams)
    rom.fit(X_train.T, Y_train)
    rom.print_summary()

    ##################################################
    # Predict validation data
    ##################################################

    X_pod = rom.predict(Y_test).T

    errors = np.zeros(len(X_pod))
    for i, (x_pod, x_test) in enumerate(zip(X_pod, X_test)):
        errors[i] = norm(x_pod - x_test) / norm(x_test)

    argmax = np.argmax(errors)
    if case == 0:
        x_pod = training.unstack_simulation_vector(X_pod[argmax])[0]
        x_test = training.unstack_simulation_vector(X_test[argmax])[0]

    print("Error Statistics:")
    print(f"Mean Error  :\t{np.mean(errors):.3g}")
    print(f"Max Error   :\t{np.max(errors):.3g}")
    print(f"Min Error   :\t{np.min(errors):.3g}")

    # worst = norm(x_pod - x_test, axis=1) / norm(x_test, axis=1)
    # plt.figure()
    # plt.title("Worst Result")
    # plt.xlabel("Time")
    # plt.ylabel("Relative $\ell_2$ Error")
    # plt.semilogy(training.times, worst, '-*b')
    # plt.grid(True)
    # plt.show()



