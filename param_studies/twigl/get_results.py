import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, 'outputs')

try:
    if len(sys.argv) != 2:
        raise AssertionError(
            'There must be a command line argument to point to '
            'the test case.\n')

    if sys.argv[1] == 'reference':
        path = os.path.join(base, sys.argv[1])
    else:
        if int(sys.argv[1]) > len(os.listdir(base)):
            raise ValueError(
                'The provided index must match a test case.')

        path = os.path.join(base, sys.argv[1].zfill(3))
        if not os.path.isdir(path):
            raise NotADirectoryError('Invalid path to test case.')

except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

n = int(sys.argv[1])
params = np.loadtxt(base+'/params.txt')[n]

# sim.plot_flux_moments(0, 1, [0.0, 0.2, 0.5])
sim.plot_power()
plt.show()

# from rom.dmd import DMD
# X = sim.create_simulation_matrix()
# dmd = DMD(svd_rank=1.0-1.0e-12)
# dmd.fit(X, sim.times)
#
# dmd.plot_error_decay()
# dmd.plot_timestep_errors()
# plt.show()

