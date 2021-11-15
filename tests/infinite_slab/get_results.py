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
            '\t0 = Subcritical\n'
            '\t1 = Delayed Supercritical\n'
            '\t2 = Prompt Supercritical')

    arg = int(sys.argv[1])
    if arg > 2:
        raise ValueError('Unrecognized result index.')
except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, 'outputs')
if arg == 0:
    path = os.path.join(base, 'subcritical')
    times = [0.0, 1.0, 2.0]
elif arg == 1:
    path = os.path.join(base, 'delayed_supercritical')
    times = [0.0, 1.0, 4.0]
else:
    path = os.path.join(base, 'prompt_supercritical')
    times = [0.0, 0.01, 0.02]

sim = SimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, 1, times)
if arg == 2:
    from typing import List
    from matplotlib.pyplot import Axes, Line2D
    ax: Axes = plt.gca()
    lines: List[Line2D] = ax.get_lines()

    # Scale data
    lines[0].set_ydata(lines[0].get_ydata() * 1.0e2)
    lines[-1].set_ydata(lines[-1].get_ydata() * 1.0e-3)

    # Get max value, normalize
    max_val = np.max(lines[0].get_ydata())
    for line in lines:
        line.set_ydata(line.get_ydata() / max_val)

    # Modify labels
    lines[0].set_label(lines[0].get_label() + r' (x $10^{2}$)')
    lines[-1].set_label(lines[-1].get_label() + r' (x $10^{-3}$)')
    ax.legend()
plt.show()

