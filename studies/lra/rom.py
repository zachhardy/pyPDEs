import copy
import os
import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from studies.utils import *
from readers import NeutronicsDatasetReader
from pyROMs import POD, DMD, PartitionedDMD

warnings.filterwarnings('ignore')

########################################
# Parse the data
########################################
study = int(sys.argv[1])
dataset = get_data('lra', study, skip=2)
n_params = dataset.n_parameters

var = 'power_density'
interior_only = False
test_size = 0.2 if n_params == 1 or not interior_only else 0.5
tau = 1.0e-8
interp = 'rbf'
eps = 5.0 if n_params == 1 else 20.0

splits = dataset.train_test_split(variables=var,
                                  test_size=test_size, seed=12,
                                  interior_only=interior_only)
X_train, X_test, Y_train, Y_test = splits


########################################
# Run the ROM
########################################
# Construct POD model, predict test data
start_time = time.time()
pod = POD(svd_rank=1.0-tau)
pod.fit(X_train, Y_train, interp, epsilon=eps)
contruction_time = time.time() - start_time
pod.print_summary()
pod.plot_singular_values()
plt.show()

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

# # Apply DMD to POD results
# X_dmd = []
# dmd_errors = np.zeros(len(X_pod))
# dmd_worst: DMD = None
# avg_dmd_time = 0.0
# for i in range(len(X_pod)):
#     t_start = time.time()
#     dmd = PartitionedDMD(DMD(svd_rank=tau, opt=True), [10, 21])
#     dmd.fit(np.array(X_pod[i], complex))
#     avg_dmd_time += (time.time()-t_start)/len(X_pod)
#
#     X_dmd.append(dmd.reconstructed_data)
#     dmd_errors[i] = norm(X_test[i]-X_dmd[i])/norm(X_dmd[i])
#     if i == np.argmax(pod_errors):
#         dmd_worst = copy.deepcopy(dmd)

# from pydmd import DMD as PyDMD
# from pydmd import MrDMD
# for i in range(len(X_pod)):
#     t_start = time.time()
#     dmd = MrDMD(PyDMD(opt=True), max_level=2)
#     dmd.fit(np.array(X_pod[i].T, dtype=complex))
#     avg_dmd_time += (time.time()-t_start)/len(X_pod)
#     X_dmd.append(dmd.reconstructed_data.T)
#     dmd_errors[i] = norm(X_test[i]-X_dmd[i])/norm(X_dmd[i])
#     if i == np.argmax(pod_errors):
#         dmd_worst = copy.deepcopy(dmd)

# Find worst POD prediction
argmax = np.argmax(pod_errors)
x_pod, x_test = X_pod[argmax], X_test[argmax]
# x_pod, x_dmd, x_test = X_pod[argmax], X_dmd[argmax], X_test[argmax]
pod_step_errors = norm(x_test-x_pod, axis=1)/norm(x_test, axis=1)
# dmd_step_errors = norm(x_test-x_dmd, axis=1)/norm(x_test, axis=1)

plt.figure()
plt.title(f"Worst Result\nError = {pod_errors[argmax]:.3e}")
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Relative $L^2$ Error", fontsize=12)
plt.semilogy(dataset.times, pod_step_errors, '-*b')
plt.grid(True)

# try:
#     reconstruction = dmd_worst.snapshot_errors
# except:
#     x = dmd_worst.snapshots.T
#     x_dmd = dmd_worst.reconstructed_data.T
#     reconstruction = norm(x - x_dmd, axis=1) / norm(x, axis=1)
#
# plt.figure()
# plt.xlabel("Time (s)", fontsize=12)
# plt.ylabel("Relative $L^2$ Error", fontsize=12)
# plt.semilogy(dataset.times, pod_step_errors, '-b*', label="POD")
# plt.semilogy(dataset.times, dmd_step_errors, '-ro', label="DMD")
# plt.semilogy(dataset.times, reconstruction, '-k+', label="DMD Reconstruction")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# Print aggregated POD results
msg = f"===== Summary of {len(pod_errors)} POD Queries ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Number of Snapshots:\t{len(X_train)}")
print(f"Number of POD Modes:\t{pod.n_modes}")
print(f"Average POD Error:\t{np.mean(pod_errors):.3e}")
print(f"Maximum POD Error:\t{np.max(pod_errors):.3e}")
print(f"Minimum POD Error:\t{np.min(pod_errors):.3e}")

# # Print DMD results
# msg = f"===== Summary of {len(pod_errors)} DMD Models ====="
# header = "=" * len(msg)
# print("\n".join(["", header, msg, header]))
# print(f"Number of Snapshots:\t{len(X_pod[0])}")
# try:
#     print(f"Number of DMD Modes:\t{dmd_worst.n_modes}")
# except:
#     print(f"Number of DMD Modes:\t{dmd_worst.modes.shape[1]}")
# print(f"Average DMD Error:\t{np.mean(dmd_errors):.3e}")
# print(f"Maximum DMD Error:\t{np.max(dmd_errors):.3e}")
# print(f"Minimum DMD Error:\t{np.min(dmd_errors):.3e}")
#
# # Print cost summary
# avg_cost = avg_predict_time + avg_dmd_time
# msg = f"===== Summary of Computational Cost ====="
# header = "=" * len(msg)
# print("\n".join(["", header, msg, header]))
# print(f"Construction Time:\t{contruction_time:.3e} s")
# print(f"Query Time:\t\t{avg_predict_time:.3e} s")
# print(f"DMD Time:\t\t{avg_dmd_time:.3e} s")
# print(f"Total Prediction Time:\t{avg_cost:.3e} s")

plt.show()
