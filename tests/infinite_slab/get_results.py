import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')

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
    times = [0.0, 1.0, 2.0]
else:
    path = os.path.join(base, 'prompt_supercritical')
    times = [0.0, 0.01, 0.02]

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, [0, 1], times, grouping='time')
for g, ax in enumerate(plt.gcf().get_axes()):
    ax.set_title(f"Group {g}")

if arg == 2:
    from typing import List
    from matplotlib.pyplot import Axes, Line2D
    for ax in plt.gcf().get_axes():
        ax: Axes = ax
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

sim.plot_power()
ax = plt.gca()
ax.set_xlabel(f"Time (s)", fontsize=12)
ax.set_ylabel(f"Power (W)", fontsize=12)
plt.show()


from pyROMs.dmd import DMD, PartitionedDMD
from pydmd import DMD as PyDMD
from pydmd import MrDMD

X = sim.create_simulation_matrix('power_density')
times = sim.times

dmd = PartitionedDMD(DMD(svd_rank=1.0e-8, opt=True), [13, 26])
dmd.fit(X)
dmd.print_summary()
dmd.print_partition_summaries(skip_line=True)

plt.figure()
snapshot_errors = dmd.snapshot_errors
plt.title(f"Partitioned DMD", fontsize=12)
plt.xlabel(f"Time (s)", fontsize=12)
plt.ylabel(f"Relative $L^2$ Error", fontsize=12)
plt.semilogy(times, snapshot_errors, '-*b')
plt.grid(True)

pydmd = MrDMD(PyDMD(opt=True), max_level=1, max_cycles=1)
pydmd.fit(np.array(X, dtype=complex).T)
X_dmd = pydmd.reconstructed_data.T
print(f"\nMrDMD Modes:\t{pydmd.modes.shape[1]}\n")


plt.figure()
mrdmd_errors = norm(X-X_dmd, axis=1)/norm(X,axis=1)
plt.title(f"MrDMD", fontsize=12)
plt.xlabel(f"Time (s)", fontsize=12)
plt.ylabel(f"Relative $L^2$ Error", fontsize=12)
plt.semilogy(times, mrdmd_errors, '-*b')
plt.grid(True)

plt.show()
