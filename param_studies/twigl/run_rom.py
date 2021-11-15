import os.path
import sys
import time

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from dataset_reader import DatasetReader
from rom.pod import POD
from rom.dmd import DMD


script_path = os.path.dirname(os.path.abspath(__file__))

# Parse the database
dataset = DatasetReader(f'{script_path}/outputs')
dataset.read_dataset()

# Get the domain information
n_groups = dataset.n_groups
grid = np.array([p.z for p in dataset.nodes])
times = dataset.times

X = dataset.create_dataset_matrix()
Y = dataset.parameters

# # Get parameters and index for reference
y_ref = [0.97666]
n_parameters = dataset.n_parameters
if dataset.n_parameters == 1:
    if y_ref in Y:
        iref = list(np.ravel(Y)).index(y_ref[0])

# Get parameter bounds
bounds = np.zeros((n_parameters, 2))
for p in range(n_parameters):
    bounds[p] = [min(Y[:, p]), max(Y[:, p])]

# Determine interior and boundary indices
interior, bndry = [], []
for i in range(len(Y)):
    y = Y[i]

    on_bndry = False
    for p in range(n_parameters):
        if any([y[p] == bounds[p][m] for m in [0, 1]]):
            on_bndry = True
            break

    if not on_bndry:
        interior += [i]
    else:
        bndry += [i]

splits = train_test_split(X[interior], Y[interior], train_size=0.75)
X_train, X_test, Y_train, Y_test = splits
X_train = np.vstack((X_train, X[bndry]))
Y_train = np.vstack((Y_train, Y[bndry]))

# Construct POD model, predict test data
tstart = time.time()
svd_rank = 1.0 - 1.0e-12
pod = POD(svd_rank=svd_rank)
pod.fit(X_train, Y_train, verbose=True)
offline_time = time.time() - tstart

tstart = time.time()
X_pred = pod.predict(Y_test, 'cubic')
predict_time = time.time() - tstart

# Format POD predictions for DMD
X_pred = dataset.unstack_simulation_vector(X_pred)
X_test = dataset.unstack_simulation_vector(X_test)

errors = []
for i in range(len(X_test)):
    error = norm(X_test[i]-X_pred[i]) / norm(X_test[i])
    errors.append(error)

argmax = np.argmax(errors)
x_pred, x_test = X_pred[argmax], X_test[argmax]

timestep_errors = []
for t in range(len(x_test)):
    error = norm(x_test[t]-x_pred[t]) / norm(x_test[t])
    timestep_errors.append(error)

# Print aggregated DMD results
msg = f'===== Summary of {len(errors)} POD Models ====='
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
print(f'Average POD Reconstruction Error:\t{np.mean(errors):.3e}')
print(f'Maximum POD Reconstruction Error:\t{np.max(errors):.3e}')
print(f'Minimum POD Reconstruction Error:\t{np.min(errors):.3e}')
print()

plt.figure()
plt.xlabel(r'Time [$\mu$s]', fontsize=12)
plt.ylabel('Relative Error [arb. units]', fontsize=12)
plt.semilogy(times, timestep_errors, '-*b')
plt.grid(True)
plt.show()

# # Construct DMD models, compute errors
# errors = np.zeros(len(X_pred))
# dmd_time = 0.0
# for i in range(len(X_pred)):
#     tstart = time.time()
#     dmd = DMD(svd_rank=svd_rank)
#     dmd.fit(X_test[i], times, verbose=False)
#     dmd_time += time.time() - tstart
#
#     x_dmd = dmd.reconstructed_data.real
#     errors[i] = norm(X_pred[i] - x_dmd) / norm(X_test[i])
# query_time = predict_time + dmd_time
#
# # Print aggregated DMD results
# msg = f'===== Summary of {errors.size} DMD Models ====='
# header = '=' * len(msg)
# print('\n'.join(['', header, msg, header]))
# print(f'Average DMD Reconstruction Error:\t{np.mean(errors):.3e}')
# print(f'Maximum DMD Reconstruction Error:\t{np.max(errors):.3e}')
# print(f'Minimum DMD Reconstruction Error:\t{np.min(errors):.3e}')
# print()
#
# msg = f'===== Summary of POD-DMD Model Cost ====='
# header = '=' * len(msg)
# print('\n'.join([header, msg, header]))
# print(f'Construction:\t\t\t{offline_time:.3e} s')
# print(f'Prediction:\t\t\t{predict_time:.3e} s')
# print(f'Decomposition:\t\t\t{dmd_time:.3e} s')
# print(f'Total query cost:\t\t{query_time:.3e} s')
# print()

plt.show()
