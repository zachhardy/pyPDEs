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

from rom.dmd import DMD
from numpy.linalg import norm

X = sim.create_simulation_matrix().T
grid = [node.z for node in sim.nodes]
times = sim.times
t0, tf, dt = times[0], times[-1], times[1]-times[0]

dmd = DMD(svd_rank=5, sort_method='amps')
dmd.snapshot_time = {'t0': t0, 'tf': tf, 'dt': dt}
dmd.fit(X)

dmd.plot_modes_1D(x=grid, imaginary=True)
dmd.plot_dynamics(t=times, logscale=False)
dmd.plot_timestep_errors()
dmd.plot_rankwise_errors()
dmd.plot_eigs()

plt.show()
