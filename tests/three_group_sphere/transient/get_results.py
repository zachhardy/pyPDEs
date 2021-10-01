import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from simulation_reader import SimulationReader

try:
    if len(sys.argv) != 2:
        raise AssertionError(
            "There must be a command line argument to point to "
            "the test case.\n"
            "Options are:\n "
            "\t0 = Finite Volume\n"
            "\t1 = Piecewise Continuous")

    arg = int(sys.argv[1])
    if arg > 1:
        raise ValueError("Unrecognized result index.")
except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, "outputs")
if arg == 0:
    path = os.path.join(base, "fv")
else:
    path = os.path.join(base, "pwc")

sim = SimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, [0, 1], [0.0, 0.05, 0.1])
plt.show()
