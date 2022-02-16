import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader

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

sim.plot_flux_moments(0, [0, 1], [0.0, 0.5], grouping='group')
plt.gcf().suptitle("")
for g, ax in enumerate(plt.gcf().get_axes()[::2]):
    ax.set_title(f"Group {g}", fontsize=12)

sim.plot_power()
plt.show()
