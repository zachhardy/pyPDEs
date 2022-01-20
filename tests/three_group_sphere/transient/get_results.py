import os
import sys
import numpy as np
import matplotlib.pyplot as plt

########################################
# Parse the command line
########################################
if len(sys.argv) != 2:
    raise AssertionError(
        'There must be a command line argument to point to '
        'the test case.\n'
        'Options for the test case are:\n '
        '\t0 = Finite Volume\n'
        '\t1 = Piecewise Continuous')

arg = int(sys.argv[1])
if arg > 1:
    raise ValueError('Unrecognized result index.')

base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, 'outputs')
if arg == 0:
    path = os.path.join(path, 'fv')
else:
    path = os.path.join(path, 'pwc')

########################################
# Get the data
########################################
from readers import NeutronicsSimulationReader
sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

times = [0.0, sim.times[-1]]
sim.plot_flux_moments(0, times=times)
plt.show()
