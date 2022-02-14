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

idx = len(X)//4 + 1
dmd = DMD(svd_rank=10).fit(X[:idx])
dmd.print_summary()

recon_error = dmd.snapshot_errors

dmd.dmd_time["tend"] = 50
X_dmd = dmd.reconstructed_data
step_errors = norm(X - X_dmd, axis=1) / norm(X, axis=1)

plt.figure()
plt.xlabel("Time ($\mu$s)", fontsize=12)
plt.ylabel("Relative $L^2$ Error", fontsize=12)
plt.semilogy(times[:idx], recon_error, '-*b',
             ms=3.0, label=f"Reconstruction")
plt.semilogy(times[idx:], step_errors[idx:], '-ro',
             ms=3.0, label=f"Extrapolation")
plt.grid(True)
plt.legend()
plt.show()
