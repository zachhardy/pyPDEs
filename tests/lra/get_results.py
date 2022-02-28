import os
import sys
import time
import warnings

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')

script_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_path, 'outputs')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

# sim.plot_flux_moments(0, [0, 1], [1.44], grouping='time')
# plt.gcf().suptitle("")
# for i, ax in enumerate(plt.gcf().get_axes()[::2]):
#     ylabel = "Y (cm)" if i == 0 else ""
#     ax.set_title(f"Group {i}", fontsize=12)
#     ax.set_xlabel("X (cm)", fontsize=12)
#     ax.set_ylabel(ylabel, fontsize=12)
# sim.plot_power(mode=1, log_scale=True)
# sim.plot_temperatures()
# plt.show()

# Get the data
X = sim.create_simulation_matrix('power_density')
times = sim.times

from pyROMs import DMD, PartitionedDMD
from numpy.linalg import norm

# Traditional DMD
dmd = DMD(svd_rank=1.0e-10, opt=False)
dmd.fit(X)

X_dmd = dmd.reconstructed_data.real
dmd_reconstruction_error = norm(X - X_dmd) / norm(X)
dmd_errors = norm(X - X_dmd, axis=1) / norm(X, axis=1)
print(f'Reconstruction Error:\t{dmd_reconstruction_error:.3e}')

# Partitioned DMD
partition_points = np.arange(30, 300, 12)
sub_dmd = DMD(svd_rank=1.0e-10, opt=True)
pdmd = PartitionedDMD(sub_dmd, partition_points)
pdmd.fit(X)

X_dmd = pdmd.reconstructed_data.real
pdmd_reconstruction_error = norm(X - X_dmd) / norm(X)
pdmd_errors = norm(X - X_dmd, axis=1) / norm(X, axis=1)
print(f'Reconstruction Error:\t{pdmd_reconstruction_error:.3e}')

# Multi-Resolution DMD
from pydmd import DMD as PyDMD
from pydmd import MrDMD
sub_dmd = PyDMD(svd_rank=-1, opt=True)
mrdmd = MrDMD(sub_dmd, max_level=3)
mrdmd.fit(np.array(X.T, dtype=complex))

X_dmd = mrdmd.reconstructed_data.real.T
mrdmd_reconstruction_error = norm(X-X_dmd)/norm(X)
mrdmd_errors = norm(X-X_dmd, axis=1)/norm(X, axis=1)
print(f'Reconstruction Error:\t{mrdmd_reconstruction_error:.3e}')

# Plotting
plt.figure()
plt.semilogy(times, dmd_errors, '-*b', ms=4.0,
             label=f"DMD: {dmd.n_modes} Modes")
plt.semilogy(times, pdmd_errors, '-or', ms=4.0,
             label=f"PDMD: {sum(pdmd.n_modes)} Modes")
plt.semilogy(times, mrdmd_errors, '-+k', ms=4.0,
             label=f"MRDMD: {mrdmd.modes.shape[1]} Modes")
plt.legend()
plt.grid(True)
plt.show()

#
# P, P_dmd = np.sum(X, axis=1), np.sum(X_dmd, axis=1)
# plt.plot(times, P, '-b', label='Simulation')
# plt.plot(times, P_dmd, '--+r', markersize=4.0, label='MrDMD')
# plt.show()
#
