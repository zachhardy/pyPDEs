import os
import numpy as np
import matplotlib.pyplot as plt

########################################
# Get the data
########################################
from readers import NeutronicsSimulationReader
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, 'outputs', 'fv')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

X = sim.create_simulation_matrix()
times = sim.times

########################################
# Show SVD thresholds
########################################
from pyROMs import POD

pod = POD(svd_rank=1.0 - 1.0e-8).fit(X, times)
svals = pod.singular_values
svals = svals/max(svals)

plt.figure()
plt.xlabel('n', fontsize=12)
plt.ylabel(r'Relative Singular Value', fontsize=12)
plt.semilogy(svals, '-*b', label='Singular Values')
plt.axhline(svals[pod.n_modes-1], xmin=0, xmax=len(svals)-1,
            color='r', label=f'$\\tau$ = {1.0-pod.svd_rank:.2e}')


pod.fit(X, times, svd_rank=0)
plt.axhline(svals[pod.n_modes-1], xmin=0, xmax=len(svals)-1,
            color='g', label=f'Optimal Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/Users/zacharyhardy/Documents/phd/dissertation/dissertation/'
            'figures/chapter2/scree_truncation.pdf')
plt.show()

########################################
# Show reconstruction errors
########################################
pod = POD(svd_rank=1)

p = -np.arange(1, 18, 1)
errors_tau = np.zeros(len(p))
for i in range(len(p)):
    pod.fit(X, times, svd_rank=1.0 - 10.0**p[i])
    errors_tau[i] = pod.reconstruction_error

x = list(range(1, len(X)+1))
errors_rank = np.zeros(len(X))
for i in range(len(X)):
    pod.fit(X, times, svd_rank=x[i])
    errors_rank[i] = pod.reconstruction_error

svals = pod.singular_values
svals = svals / max(svals)

from matplotlib.pyplot import Axes
plt.figure()
ax: Axes = plt.gca()
ax.set_xlabel('# of Modes', fontsize=12)
ax.set_ylabel('Relative  $L^2$  Error', fontsize=12)
ln_s = ax.semilogy(x, svals, '-*b', label='Singular Values')
ln_n = ax.semilogy(x, errors_rank, '-ro', label='Rank')

axtwin: Axes = ax.twiny()
axtwin.set_xlabel(r'$-\log_{10}(\tau)$', fontsize=12)
axtwin.set_xticks(-p[::2])
ln_tau = axtwin.semilogy(-p, errors_tau, '-+g',
                         label=r'Energy Retention Limit ($\tau$)')

lines = ln_s + ln_n + ln_tau
labels = [line.get_label() for line in lines]

ax.legend(lines, labels)
ax.grid(True)
plt.savefig('/Users/zacharyhardy/Documents/phd/dissertation/dissertation/'
            'figures/chapter2/pod_error_decay.pdf')
plt.show()
