import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from simulation_reader import SimulationReader


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, 'outputs')

try:
    if len(sys.argv) != 2:
        raise AssertionError(
            'There must be a command line argument to point to '
            'the test case.\n')


    if int(sys.argv[1]) > len(os.listdir(base)):
        raise ValueError('The provided index must match a test case.')

    path = os.path.join(base, sys.argv[1].zfill(3))
    if not os.path.isdir(path):
        raise NotADirectoryError('Invalid path to test case.')

except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


sim = SimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, 0, [0.0, 0.6, 1.1, 2.0])
sim.plot_power()
plt.show()


from rom.dmd import DMD
X = sim.create_simulation_matrix()
dmd = DMD(svd_rank=1.0-1.0e-12)
dmd.fit(X, sim.times)

dmd.plot_singular_values()
dmd.plot_error_decay()
dmd.plot_timestep_errors()
plt.show()
