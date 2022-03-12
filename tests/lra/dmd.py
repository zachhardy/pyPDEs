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

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, 'outputs')
print(path)

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

X = sim.create_simulation_matrix('power_density')
times = sim.times

dmd = DMD(svd_rank=1.0e-10, opt=True).fit(X)
dmd.print_summary()
dmd_errors = dmd.snapshot_errors

partition_points = list(range(12, 300, 12))
pdmd = PartitionedDMD(DMD(svd_rank=1.0e-10, opt=True), partition_points)
pdmd.fit(X)
pdmd_errors = pdmd.snapshot_errors

mrdmd = MrDMD(PyDMD(opt=True), max_level=5, max_cycles=1)
mrdmd.fit(np.array(X, dtype=complex).T)
X_dmd = mrdmd.reconstructed_data.T
mrdmd_reconstruction_error = norm(X-X_dmd)/norm(X)
mrdmd_errors = norm(X-X_dmd, axis=1)/norm(X, axis=1)

plt.figure()
plt.tick_params(labelsize=12)
snapshot_errors = pdmd.snapshot_errors
plt.xlabel(f"Time (s)", fontsize=12)
plt.ylabel(f"Relative $L^2$ Error", fontsize=12)
plt.semilogy(times, dmd_errors, '-*b',
             ms=4.0, label=f"DMD: {dmd.n_modes} Modes")
plt.semilogy(times, pdmd_errors, '-or',
             ms=4.0, label=f"Partitioned DMD: {sum(pdmd.n_modes)} Modes")
plt.semilogy(times, mrdmd_errors, '-+k',
             ms=4.0, label=f"MrDMD: {mrdmd.modes.shape[1]} Modes")
plt.legend()
plt.grid(True)

print("===== Comparison of DMD Methods =====")
print(f"# of Modes:\n"
      f"\t{'DMD:':<20}\t{dmd.n_modes}\n"
      f"\t{'Partitioned DMD:':<20}\t{sum(pdmd.n_modes)}\n"
      f"\t{'MRDMD:':<20}\t{len(mrdmd.modes.T)}")
print(f"Reconstruction Error:\n"
      f"\t{'DMD:':<20}\t{dmd.reconstruction_error:.3e}\n"
      f"\t{'Partitioned DMD:':<20}\t{pdmd.reconstruction_error:.3e}\n"
      f"\t{'MRDMD:':<20}\t{mrdmd_reconstruction_error:.3e}")
print(f"Mean Snapshot Error:\n"
      f"\t{'DMD:':<20}\t{np.mean(dmd_errors):.3e}\n"
      f"\t{'Partitioned DMD:':<20}\t{np.mean(pdmd_errors):.3e}\n"
      f"\t{'MRDMD:':<20}\t{np.mean(mrdmd_errors):.3e}")
print(f"Max Snapshot Error:\n"
      f"\t{'DMD:':<20}\t{np.max(dmd_errors):.3e}\n"
      f"\t{'Partitioned DMD:':<20}\t{np.max(pdmd_errors):.3e}\n"
      f"\t{'MRDMD:':<20}\t{np.max(mrdmd_errors):.3e}")

plt.show()