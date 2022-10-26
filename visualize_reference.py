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
    if problem_name not in ["Sphere3g", "ReflectedSphere", "LRA"]:
        raise ValueError(f"{problem_name} is not a valid problem.")

    save = False
    for arg in sys.argv[2:]:
        argval = arg.split("=")[1]
        if "save=" in arg:
            save = bool(int(argval))

    path = os.path.abspath(os.path.dirname(__file__))
    data_dir = f"{path}/Problems/{problem_name}/reference"
    outpath = "/Users/zhardy/projects/POD-MCI/papers/journal"
    outpath = f"{outpath}/figures/{problem_name}/ref"
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    reader = NeutronicsSimulationReader(data_dir).read()

    if problem_name == "Sphere3g":
        fname = f"{outdir}/sflux" if save else None
        reader.plot_flux_moment(0, -1, [0.0, reader.times[-1]], filename=fname)

        fname = f"{outdir}/power" if save else None
        reader.plot_power(filename=fname)

    if problem_name == "LRA":
        fname = f"{outdir}/sflux" if save else None
        reader.plot_flux_moment(0, [0, 1], [0.0, 1.44], filename=fname)

        fname = f"{outdir}/power_profile" if save else None
        reader.plot_power("BOTH", True, filename=fname)

        fname = f"{outdir}/temperature_profile" if save else None
        reader.plot_fuel_temperature("BOTH", False, filename=fname)

    if problem_name == "ReflectedSphere":
        reader.plot_flux_moment(0, -1, [0.0, reader.times[-1]])
        reader.plot_power()
    plt.show()
