import os
import sys
import warnings
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from pyROMs import DMD
from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')

########################################
# Get the data
########################################
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, 'outputs', 'fv')
sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

r = [p.z for p in sim.nodes]
times = sim.times
X = sim.create_simulation_matrix()

train = 101
stop = 501

dmd = DMD(svd_rank=1.0e-8).fit(X[:train])
dmd.print_summary()
recon_error = dmd.snapshot_errors

dmd.dmd_time["tend"] = stop - 1
X_dmd = dmd.reconstructed_data
step_errors = norm(X[:stop] - X_dmd, axis=1) / norm(X[:stop], axis=1)

plt.figure()
plt.xlabel("Time ($\mu$s)", fontsize=12)
plt.ylabel("Relative $L^2$ Error", fontsize=12)
plt.semilogy(times[:train], recon_error, '-*b',
             ms=3.0, label=f"Reconstruction")
plt.semilogy(times[train:stop:4], step_errors[train:stop:4], '-or',
             ms=3.0, label=f"Extrapolation")
plt.grid(True)
plt.legend()
plt.show()
