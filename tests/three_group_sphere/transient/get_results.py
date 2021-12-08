import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader

try:
    if len(sys.argv) != 2:
        raise AssertionError(
            'There must be a command line argument to point to '
            'the test case.\n'
            'Options are:\n '
            '\t0 = Finite Volume\n'
            '\t1 = Piecewise Continuous')

    arg = int(sys.argv[1])
    if arg > 1:
        raise ValueError('Unrecognized result index.')
except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()


script_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_path, 'outputs')
if arg == 0:
    path = os.path.join(base, 'fv')
else:
    path = os.path.join(base, 'pwc')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

from rom.dmd import DMD
from numpy.linalg import norm

X = sim.create_simulation_matrix().T
grid = [node.z for node in sim.nodes]
times = sim.times
t0, tf, dt = times[0], times[-1], times[1]-times[0]

n = 4

dmd = DMD(svd_rank=n, sort_method='amps')
dmd.snapshot_time = {'t0': t0, 'tf': tf, 'dt': dt}
dmd.fit(X)

Xdmd = dmd.reconstructed_data

step_errors = []
for t in range(len(times)):
    x, xdmd = norm(X[:, t]), norm(Xdmd[:, t])
    error = norm(x - xdmd) / norm(x)
    step_errors.append(error)

plt.figure()
plt.xlabel('Time [sec]', fontsize=14)
plt.ylabel(r'Relative $L^2$ Error', fontsize=14)
plt.semilogy(times, step_errors, 'b*-')
plt.grid(True)

# fname = '/Users/zacharyhardy/Documents/proposal/' \
#         'revision1/figures/dmd_timestep_errors.pdf'
# plt.savefig(fname)
plt.show()

from modules.neutron_diffusion.analytic import *

exact: AnalyticSolution = load(script_path + '/sphere5cm.obj')
alphas = np.array([mode.alpha for mode in exact.modes[:n]])
alphas = np.exp(alphas * dt)

eigs = dmd.eigenvalues
for i in range(len(eigs)):
    if eigs[i].imag != 0.0:
        omega = np.log(eigs[i])/dt
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

# fname = '/Users/zacharyhardy/Documents/proposal/' \
#         'revision1/figures/dmd_eigenvalues.pdf'
# plt.savefig(fname)
plt.show()
