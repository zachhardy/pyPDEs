import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from simulation_reader import SimulationReader

try:
    if len(sys.argv) != 2:
        raise AssertionError(
            'There must be a command line argument to point to '
            'the test case.\n'
            'Options are:\n '
            '\t0 = Ramp Reactivity Increase\n'
            '\t1 = Step Reactivity Increase')

    arg = int(sys.argv[1])
    if arg > 1:
        raise ValueError('Unrecognized result index.')
except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, 'outputs')
if arg == 0:
    path = os.path.join(base, 'ramp')
else:
    path = os.path.join(base, 'step')

sim = SimulationReader(path)
sim.read_simulation_data()

# sim.plot_flux_moments(0, 0, [0.0, 0.2, 0.5])
# sim.plot_power()
# plt.show()

from rom.dmd import DMD
from rom.pod import POD
X = sim.create_simulation_matrix().T
t = sim.times

dmd = DMD(svd_rank=10)
dmd.fit(X)
# dmd.plot_scree()
# dmd.plot_rankwise_errors()

pod = POD(svd_rank=10)
pod.fit(X, t)
pod.plot_scree()
pod.plot_rankwise_errors()

plt.show()
