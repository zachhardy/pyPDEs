import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from os.path import splitext

from readers import NeutronicsSimulationReader


if __name__ == "__main__":

    if len(sys.argv) < 2:
        msg = "Invalid command line arguments. "
        msg += "A problem name must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    if problem_name not in ["Sphere3g", "InfiniteSlab", "TWIGL", "LRA"]:
        raise ValueError(f"{problem_name} is not a valid problem.")

    path = os.path.abspath(os.path.dirname(__file__))
    data_path = f"{path}/Problems/{problem_name}/outputs"
    outpath = "/Users/zhardy/Documents/Journal Papers/POD-MCI/figures"

    reader = NeutronicsSimulationReader(data_path).read()

    if problem_name == "Sphere3g":
        fname = f"{outpath}/Sphere3g/ref/sflux"
        reader.plot_flux_moment(0, -1, [0.0, reader.times[-1]], filename=fname)

        fname = f"{outpath}/Sphere3g/ref/power"
        reader.plot_power(filename=fname)

    plt.show()
