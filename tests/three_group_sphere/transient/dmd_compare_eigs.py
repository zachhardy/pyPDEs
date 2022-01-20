import os
import numpy as np
import matplotlib.pyplot as plt

########################################
# Get the data
########################################
from readers import NeutronicsSimulationReader
base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, 'outputs', 'fv')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

X = sim.create_simulation_matrix()
times = sim.times

########################################
# Perform DMD
########################################
from pyROMs import DMD
dmd = DMD(svd_rank=4).fit(X)
dmd.print_summary()

########################################
# Compare to exact eigenvalues
########################################
from modules.neutron_diffusion.analytic import *

exact: AnalyticSolution = load(base + '/sphere6cm.obj')
alphas = [exact.get_mode(i, 0, 'amp').alpha for i in range(dmd.n_modes)]
alphas = np.exp(np.array(alphas) * (times[1]-times[0]))

eigs = dmd.eigvals
for i in range(len(eigs)):
    if eigs[i].imag != 0.0:
        omega = np.log(eigs[i])/(times[1]-times[0])
        if omega.imag % np.pi < 1.0e-12:
            eigs[i] = eigs[i].real + 0.0j

plt.figure()
plt.xlabel(r'$\mathcal{R}~(\lambda)$', fontsize=14)
plt.ylabel(r'$\mathcal{I}~(\lambda)$', fontsize=14)
plt.plot(eigs.real, eigs.imag, 'bo', label='DMD')
plt.plot(alphas.real, alphas.imag, 'r*', label='Exact')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
