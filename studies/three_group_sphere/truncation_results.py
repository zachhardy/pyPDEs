import os
import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from studies.utils import *
from pyROMs.pod import POD

warnings.filterwarnings('ignore')

########################################
# Parse the data
########################################
study = int(sys.argv[1])
dataset = get_data('three_group_sphere', study)

var = 'power_density'
test_size = 0.2 if study == 0 else 0.5
interior_only = False
tau = 1.0e-8
interp = 'rbf_gaussian' if study == 0 else 'rbf_cubic'
eps = 1.0 if study == 0 else 200.0

if study == 3 and not interior_only:
    interp = 'rbf'
    test_size = 0.25

splits = dataset.train_test_split(variables=var,
                                  test_size=test_size, seed=12,
                                  interior_only=interior_only)
X_train, X_test, Y_train, Y_test = splits

rom_info = {'tau': [], 'n_modes': [],
            'mean_error': [], 'max_error': [],
            'min_error': [], 'predict_time': []}

taus = [10.0**i for i in range(-16, 0)]
for tau in taus:

    # Construct POD model, predict test data
    start_time = time.time()
    svd_rank = 1.0 - tau
    pod = POD(svd_rank=svd_rank)
    pod.fit(X_train, Y_train, interp, epsilon=eps)
    contruction_time = time.time() - start_time

    # Predict results
    X_pred = []
    avg_predict_time = 0.0
    for y_test in Y_test:
        start_time = time.time()
        X_pred.append(pod.predict(y_test))
        avg_predict_time += time.time() - start_time
    avg_predict_time /= len(Y_test)

    # Format datasets
    X_pred = dataset.unstack_simulation_vector(X_pred)
    X_test = dataset.unstack_simulation_vector(X_test)

    # Compute simulation errors
    errors = np.zeros(len(X_test))
    for i in range(len(X_test)):
        errors[i] = norm(X_test[i]-X_pred[i])/norm(X_test[i])

    rom_info['tau'].append(tau)
    rom_info['n_modes'].append(pod.n_modes)
    rom_info['predict_time'].append(avg_predict_time)
    rom_info['mean_error'].append(np.mean(errors))
    rom_info['max_error'].append(np.max(errors))
    rom_info['min_error'].append(np.min(errors))

print(f"Training set size:\t{len(X_train)}")

from typing import List
from matplotlib.pyplot import Figure, Axes

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig: Figure = fig
axs: List[Axes] = axs.ravel()

for i, ax in enumerate(axs):
    if i == 0:
        ax.set_xlabel(f"# of Modes", fontsize=12)
        ax.set_ylabel(f"Relative $L^2$ Error", fontsize=12)
        ax.semilogy(rom_info['n_modes'], rom_info['mean_error'],
                    '-b*', label="Mean Error")
        ax.semilogy(rom_info['n_modes'], rom_info['max_error'],
                    '-ro', label="Max Error")
        ax.semilogy(rom_info['n_modes'], rom_info['min_error'],
                    '-k+', label="Min Error")
        ax.legend()
        ax.grid(True)
    else:
        ax.set_xlabel(f"$\\tau$", fontsize=12)
        ax.loglog(taus, rom_info['mean_error'], '-b*', label="Mean Error")
        ax.loglog(taus, rom_info['max_error'], '-ro', label="Max Error")
        ax.loglog(taus, rom_info['min_error'], '-k+', label="Min Error")
        ax.legend()
        ax.grid(True)
plt.tight_layout()
plt.show()
