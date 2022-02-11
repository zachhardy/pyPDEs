import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from studies.utils import *
from pyROMs.pod import POD
from pyROMs.dmd import DMD

warnings.filterwarnings('ignore')

########################################
# Parse the data
########################################
study = int(sys.argv[1])
dataset = get_data('three_group_sphere', study)

var = 'power_density'
test_size = 0.2 if study == 0 else 0.5
interior_only = True
tau = 1.0e-8
interp = 'rbf_gaussian'
eps = 1.0 if study == 0 else 200.0

splits = dataset.train_test_split(variables=var,
                                  test_size=test_size, seed=12,
                                  interior_only=interior_only)
X_train, X_test, Y_train, Y_test = splits

# Construct POD model, predict test data
start_time = time.time()
tau = tau
pod = POD(svd_rank=1.0-tau)
pod.fit(X_train, Y_train, interp, epsilon=eps)
contruction_time = time.time() - start_time
pod.print_summary()

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

# Print aggregated POD results
msg = f"===== Summary of {len(errors)} POD Queries ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Number of Training Samples:\t\t{len(X_train)}")
print(f"Number of POD Modes:\t\t\t{pod.n_modes}")
print(f"Average POD Reconstruction Error:\t{np.mean(errors):.3e}")
print(f"Maximum POD Reconstruction Error:\t{np.max(errors):.3e}")
print(f"Minimum POD Reconstruction Error:\t{np.min(errors):.3e}")
print(f"Construction Time:\t\t\t{contruction_time:.3e} s")
print(f"Average Query Time:\t\t\t{avg_predict_time:.3e} s")

argmax = np.argmax(errors)
x_pred, x_test = X_pred[argmax], X_test[argmax]
step_errors = norm(x_test-x_pred, axis=1)/norm(x_test, axis=1)

r_b = dataset.parameters[argmax][0]
title = f"Worst Result"
if study == 0:
    title += f": $r_b = {r_b:.4f}$ cm"
title += f"\nError = {errors[argmax]:.3e}"

plt.figure()
plt.title(title, fontsize=12)
plt.xlabel("Time ($\mu$s)", fontsize=12)
plt.ylabel("Relative $L^2$ Error", fontsize=12)
plt.semilogy(dataset.times, step_errors, '-b*')
plt.grid(True)
plt.tight_layout()
# plt.show()

if dataset.n_parameters == 1:
    pod.plot_coefficients([0, 1, -2, -1])
    # plt.show()

    X_pod = pod.reconstructed_data
    X_pod = dataset.unstack_simulation_vector(X_pod)
    X_train = dataset.unstack_simulation_vector(X_train)

    modes = dataset.unstack_simulation_vector(pod.modes.T)

    from typing import List
    from matplotlib.pyplot import Figure, Axes

    grid = [p.z for p in dataset.nodes]
    n_grps = X_pod.shape[2] // len(grid)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig: Figure = fig
    axs: List[Axes] = np.ravel(axs)
    fig.suptitle("Mode Initial Conditions", fontsize=12)
    for i, ax in enumerate(axs):
        title = f"Mode {i}"
        xlabel = "r (cm)" if i > 1 else ""
        ylabel = r"$\phi_g(r)$" if i in [0, 2] else ""
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)

        mode = modes[i]
        if mode[0][np.argmax(np.abs(mode[0]))] < 0.0:
            mode *= -1.0
        for g in range(n_grps):
            ax.plot(grid, mode[0][g::n_grps], label=f"Group {g}")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()

    for m in range(4):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig: Figure = fig
        axs: List[Axes] = np.ravel(axs)

        fig.suptitle(f"Mode {m} Evolution", fontsize=12)
        times = [0.0, 0.01, 0.05, 0.1]
        mode = modes[m]
        for i, ax in enumerate(axs):
            ind = list(dataset.times).index(times[i])
            title = f"Time = {dataset.times[ind]:.2e} $\mu$s"
            xlabel = "r (cm)" if i > 1 else ""
            ylabel = r"$\phi_g(r)$" if i in [0, 2] else ""
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel(xlabel, fontsize=12)
            for g in range(n_grps):
                ax.plot(grid, mode[ind][g::n_grps], label=f"Group {g}")
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
    plt.show()
