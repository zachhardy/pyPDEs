import copy
import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes
from typing import List

from studies.utils import *
from pyROMs.pod import POD
from pyROMs.dmd import DMD

warnings.filterwarnings('ignore')


########################################
# Parse the data
########################################
problem, study = int(sys.argv[1]), int(sys.argv[2])
dataset = get_data('infinite_slab', problem, study)

var = 'power_density'
test_size = 0.2 if study in [0, 1, 2] else 0.5
interior_only = True
tau = 1.0e-8
interp = 'rbf_gaussian'
eps = 5.0
if not interior_only:
    test_size = 0.2

splits = dataset.train_test_split(variables=var,
                                  test_size=test_size, seed=12,
                                  interior_only=interior_only)
X_train, X_test, Y_train, Y_test = splits

# Construct POD model, predict test data
start_time = time.time()
pod = POD(svd_rank=1.0-tau)
pod.fit(X_train, Y_train, interp, epsilon=eps)
contruction_time = time.time() - start_time
pod.print_summary()

# Predict results
X_pod = []
avg_predict_time = 0.0
for y_test in Y_test:
    start_time = time.time()
    X_pod.append(pod.predict(y_test))
    avg_predict_time += time.time() - start_time
avg_predict_time /= len(Y_test)

# Format datasets
X_pod = dataset.unstack_simulation_vector(X_pod)
X_test = dataset.unstack_simulation_vector(X_test)
modes = dataset.unstack_simulation_vector(pod.modes.T)

# Compute POD errors
pod_errors = np.zeros(len(X_test))
for i in range(len(X_test)):
    pod_errors[i] = norm(X_test[i]-X_pod[i]) / norm(X_test[i])

# Apply DMD to POD results
X_dmd = []
dmd_errors = np.zeros(len(X_pod))
dmd_worst: DMD = None
avg_dmd_time = 0.0
for i in range(len(X_pod)):
    t_start = time.time()
    dmd = DMD(svd_rank=tau, opt=True).fit(X_pod[i])
    avg_dmd_time += (time.time()-t_start)/len(X_pod)

    X_dmd.append(dmd.reconstructed_data)
    dmd_errors[i] = norm(X_test[i]-X_dmd[i])/norm(X_dmd[i])
    if i == np.argmax(pod_errors):
        dmd_worst = copy.deepcopy(dmd)

# Find worst POD prediction
argmax = np.argmax(pod_errors)
x_pod, x_dmd, x_test = X_pod[argmax], X_dmd[argmax], X_test[argmax]
pod_step_errors = norm(x_test-x_pod, axis=1)/norm(x_test, axis=1)
dmd_step_errors = norm(x_test-x_dmd, axis=1)/norm(x_test, axis=1)

recon = []
for m in range(1, dmd_worst.n_snapshots):
    dmd = DMD(svd_rank=m, opt=True).fit(x_pod)
    recon.append(dmd.reconstruction_error)
dmd.plot_singular_values(show_rank=True)
plt.gca().semilogy(recon, '-ro')

# dmd_worst.find_optimal_parameters()
reconstruction = dmd_worst.snapshot_errors

plt.figure()
r_b = dataset.parameters[argmax][0]
plt.xlabel("Time ($\mu$s)", fontsize=12)
plt.ylabel("Relative $L^2$ Error", fontsize=12)
plt.semilogy(dataset.times, pod_step_errors, '-b*', label="POD")
plt.semilogy(dataset.times, dmd_step_errors, '-ro', label="DMD")
plt.semilogy(dataset.times, reconstruction, '-k+', label="DMD Reconstruction")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Print aggregated POD results
msg = f"===== Summary of {len(pod_errors)} POD Queries ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Number of Snapshots:\t{len(X_train)}")
print(f"Number of POD Modes:\t{pod.n_modes}")
print(f"Average POD Error:\t{np.mean(pod_errors):.3e}")
print(f"Maximum POD Error:\t{np.max(pod_errors):.3e}")
print(f"Minimum POD Error:\t{np.min(pod_errors):.3e}")

# Print DMD results
msg = f"===== Summary of {len(pod_errors)} DMD Models ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Number of Snapshots:\t{len(X_pod[0])}")
print(f"Number of DMD Modes:\t{dmd_worst.n_modes}")
print(f"Average DMD Error:\t{np.mean(dmd_errors):.3e}")
print(f"Maximum DMD Error:\t{np.max(dmd_errors):.3e}")
print(f"Minimum DMD Error:\t{np.min(dmd_errors):.3e}")

# Print cost summary
avg_cost = avg_predict_time + avg_dmd_time
msg = f"===== Summary of Computational Cost ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Construction Time:\t{contruction_time:.3e} s")
print(f"Query Time:\t\t{avg_predict_time:.3e} s")
print(f"DMD Time:\t\t{avg_dmd_time:.3e} s")
print(f"Total Prediction Time:\t{avg_cost:.3e} s")

# if dataset.n_parameters == 1:
#     pod.plot_coefficients([0, 1, -2, -1])
#     plot_mode_ics()
#     plot_mode_evoloutions([0.0, 0.01, 0.05, 0.1])

plt.show()
