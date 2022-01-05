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

# sim.plot_flux_moments(0, 0, [0.0, 1.44])
# sim.plot_power(mode=1, log_scale=True)
# sim.plot_temperatures(mode=0)
# plt.show()

# Get the data
X = sim.create_simulation_matrix('power_density')
times = sim.times

from rom.dmd import DMD, PartitionedDMD
from numpy.linalg import norm

# partition_points = [136, 150, 200]
# svd_ranks = [15, 13, 40, 0]
# opts = [True, False, False, True]
# exacts = [False, False, False, False]

partition_points = np.arange(30, 300, 30)
svd_ranks = [0]*(len(partition_points) + 1)
opts = [True]*(len(partition_points) + 1)
exacts = [False]*(len(partition_points) + 1)

options = [None]*(len(partition_points) + 1)
for i in range(len(options)):
    options[i] = {'svd_rank': svd_ranks[i],
                  'opt': opts[i],
                  'exact': exacts[i]}

sub_dmd = DMD()
dmd = PartitionedDMD(sub_dmd, partition_points, options)
dmd.fit(X.T)

dmd.find_optimal_parameters()

X_dmd = dmd.reconstructed_data.real.T
reconstruction_error = norm(X - X_dmd) / norm(X)
print(f'Reconstruction Error:\t{reconstruction_error:.3e}')

errors = norm(X - X_dmd, axis=1) / norm(X, axis=1)
argmax = np.argmax(errors)
print(f'Max Snapshot Error:\t{argmax}, {errors[argmax]:.3e}')
plt.semilogy(times, errors, '-*b')
plt.show()

P = np.sum(X, axis=1)
plt.semilogy(times, P, '-b', label='Reference')

P_dmd = np.sum(X_dmd, axis=1)
plt.semilogy(times, P_dmd, '--+r', ms=5.0, label='DMD')

plt.legend()
plt.show()

# from pydmd import DMD, MrDMD
# from numpy.linalg import norm
# sub_dmd = DMD(svd_rank=0, opt=True)
# dmd = MrDMD(sub_dmd, max_level=6)
# dmd.fit(np.array(X.T, dtype=complex))
#
# X_dmd = dmd.reconstructed_data.real.T
#
# reconstruction_error = norm(X - X_dmd) / norm(X)
# print(f'Reconstruction Error:\t{reconstruction_error:.3e}')
#
# timestep_errors = norm(X - X_dmd, axis=1) / norm(X, axis=1)
# plt.semilogy(times, timestep_errors)
# plt.show()
#
# P, P_dmd = np.sum(X, axis=1), np.sum(X_dmd, axis=1)
# plt.semilogy(times, P, '-b', label='Simulation')
# plt.semilogy(times, P_dmd, '--+r', markersize=4.0, label='MrDMD')
# plt.show()
