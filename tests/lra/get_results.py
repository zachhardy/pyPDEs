import os
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')

script_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_path, 'outputs')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, 0, [0.0, 1.44])
sim.plot_power(mode=1, log_scale=True)
sim.plot_temperatures(mode=0)
plt.show()

from rom.dmd import DMD, PartitionedDMD
from numpy.linalg import norm

# Get the data
X = sim.create_simulation_matrix(variables='power_density')
times = sim.times

partition_points = [136, 150]
svd_ranks = [10, 13, 40]
opts = [True, False, False]

options = [None] * len(svd_ranks)
for i in range(len(options)):
    options[i] = {'svd_rank': svd_ranks[i],
                  'opt': opts[i]}

dmd = DMD()
pdmd = PartitionedDMD(dmd, partition_points, options)
pdmd.fit(X.T)

Xdmd = pdmd.reconstructed_data
print(f'Reconstruction Error:\t{pdmd.reconstruction_error:.3e}')
for p in range(pdmd.n_partitions):
    print(f'\tReconstruction Error Partition {p}:\t'
          f'{pdmd.partial_reconstruction_error(p):.3e}')

errors = pdmd.snapshot_reconstruction_errors
print(f'Max Snapshot Error:\t{np.argmax(errors)}, {np.max(errors):.3e}')
plt.semilogy(times, pdmd.snapshot_reconstruction_errors)
plt.show()

# x = np.unique([node.x for node in sim.nodes])
# y = np.unique([node.y for node in sim.nodes])
# for dmd in [pdmd[0]]:
#     dmd.plot_modes_2D(x=x, y=y)
# plt.show()

P = np.sum(X, axis=1)
plt.plot(times, P, '-k', label='Reference')

Pdmd = np.sum(Xdmd.T, axis=1)
plt.plot(times, Pdmd, '*b', label='DMD')

# plt.show()
