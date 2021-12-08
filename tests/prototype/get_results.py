import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from neutronics_reader import NeutronicsReader

try:
    if len(sys.argv) != 2:
        raise AssertionError(
            'There must be a command line argument to point to '
            'the test case.\n'
            'Options are:\n '
            '\t0 = Prototype\n'
            '\t1 = Minicore')

    arg = int(sys.argv[1])
    if arg > 1:
        raise ValueError('Unrecognized result index.')
except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, 'outputs')
if arg == 0:
    path = os.path.join(base, 'prototype')
else:
    path = os.path.join(base, 'minicore')

sim = NeutronicsReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, 0, [0.0, 0.6, 1.1, 2.0])
sim.plot_power()
plt.show()


from rom.dmd import DMD
X = sim.create_simulation_matrix().T
dmd = DMD(svd_rank=1.0-1.0e-12, opt=False)
dmd.fit(X)

dmd.plot_rankwise_errors()
dmd.plot_timestep_errors()
plt.show()
