import os
import sys
import warnings
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from pyROMs import DMD
from readers import NeutronicsSimulationReader

warnings.filterwarnings('ignore')


def plot_reconstruction_errors(dmd: DMD):
    tau_error = {'mean_error': [],
                 'max_error': [],
                 'min_error': [],
                 'tau': [10.0 ** i for i in range(-18, 0)]}
    for tau in tau_error['tau']:
        dmd.fit(X, svd_rank=1.0 - tau)
        errors = dmd.snapshot_errors
        tau_error['mean_error'].append(np.mean(errors))
        tau_error['max_error'].append(np.max(errors))
        tau_error['min_error'].append(np.min(errors))

    mode_error = {'mean_error': [],
                  'max_error': [],
                  'min_error': [],
                  'n': list(range(1, len(X)))}
    for m in mode_error['n']:
        dmd.fit(X, svd_rank=m)
        errors = dmd.snapshot_errors
        mode_error['mean_error'].append(np.mean(errors))
        mode_error['max_error'].append(np.max(errors))
        mode_error['min_error'].append(np.min(errors))

    from typing import List
    from matplotlib.pyplot import Figure, Axes

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    fig: Figure = fig
    axs: List[Axes] = axs.ravel()

    for i, ax in enumerate(axs):
        if i == 0:
            n = mode_error['n']
            ax.set_xlabel(f"# of Modes", fontsize=12)
            ax.set_ylabel(f"Relative $L^2$ Error", fontsize=12)
            ax.semilogy(n, mode_error['mean_error'],
                        '-b*', label="Mean Error")
            ax.semilogy(n, mode_error['max_error'],
                        '-ro', label="Max Error")
            ax.semilogy(n, mode_error['min_error'],
                        '-k+', label="Min Error")
            ax.legend()
            ax.grid(True)
        else:
            tau = tau_error['tau']
            ax.set_xlabel(f"$\\tau$", fontsize=12)
            ax.loglog(tau, tau_error['mean_error'], '-b*', label="Mean Error")
            ax.loglog(tau, tau_error['max_error'], '-ro', label="Max Error")
            ax.loglog(tau, tau_error['min_error'], '-k+', label="Min Error")
            ax.legend()
            ax.grid(True)
    plt.tight_layout()
    plt.show()


########################################
# Parse the command line
########################################
if len(sys.argv) != 2:
    raise AssertionError(
        'There must be a command line argument to point to '
        'the test case.\n'
        'Options for the test case are:\n '
        '\t0 = Finite Volume\n'
        '\t1 = Piecewise Continuous')

arg = int(sys.argv[1])
if arg > 1:
    raise ValueError('Unrecognized result index.')

base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, 'outputs')
if arg == 0:
    path = os.path.join(path, 'fv')
else:
    path = os.path.join(path, 'pwc')

########################################
# Get the data
########################################
sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

times = sim.times
sim.plot_flux_moments(0, times=[0.0, times[-1]/2, times[-1]])
for ax in plt.gcf().get_axes():
    ax.tick_params(labelsize=12)
    ax.set_xlabel("r (cm)", fontsize=12)
    ax.set_ylabel(f"$\phi_g(r)$", fontsize=12)
    ax.set_title(ax.get_title(), fontsize=12)
    ax.legend(fontsize=12)
plt.show()
