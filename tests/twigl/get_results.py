import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from readers import NeutronicsSimulationReader
from pyROMs.dmd import DMD, PartitionedDMD
from pydmd import DMD as PyDMD
from pydmd import MrDMD

warnings.filterwarnings('ignore')

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


path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(path, 'outputs')
if arg == 0:
    path = os.path.join(base, 'ramp')
else:
    path = os.path.join(base, 'step')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()
sim.plot_power()

plt.show()
