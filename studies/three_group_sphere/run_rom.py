import os
import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from readers import NeutronicsDatasetReader
from pyROMs.pod import POD
from pyROMs.dmd import DMD

warnings.filterwarnings('ignore')

########################################
# Get the path to results
########################################
path = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 3:
    raise AssertionError(
        f'There must be a command line argument to point '
        f'to the problem type and study.')

problem = int(sys.argv[1])
if problem > 1:
    raise ValueError('Invalid problem number.')

study = int(sys.argv[2])
if study > 2:
    raise ValueError('Invalid study number')

# Get problem name
if problem == 0:
    problem_name = 'keigenvalue'
else:
    problem_name = 'ics'

# Get parameter study name
if study == 0:
    study_name = 'density'
elif study == 1:
    study_name = 'size'
else:
    study_name = 'density_size'

# Define path
path = f'{path}/outputs/{problem_name}/{study_name}'

# Check path
if not os.path.isdir(path):
    raise NotADirectoryError('Invalid path.')

########################################
# Parse the data
########################################
dataset = NeutronicsDatasetReader(path)
dataset.read_dataset()

# Get the domain information
n_groups = dataset.n_groups
grid = np.array([p.z for p in dataset.nodes])
times = dataset.times

X = dataset.create_dataset_matrix(variables='power_density')
Y = dataset.parameters

# Get number of parameters
n_parameters = dataset.n_parameters

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

    if not on_bndry:  interior += [i]
    else:  bndry += [i]

splits = train_test_split(X[interior], Y[interior], test_size=0.2)
X_train, X_test, Y_train, Y_test = splits
X_train = np.vstack((X_train, X[bndry]))
Y_train = np.vstack((Y_train, Y[bndry]))

# Construct POD model, predict test data
start_time = time.time()
svd_rank = 1.0-1.0e-12
pod = POD(svd_rank=svd_rank)
pod.fit(X_train, Y_train, 'rbf')
construction_time = time.time() - start_time

X_pred, avg_predict_time = [], 0.0
for y_test in Y_test:
    start_time = time.time()
    X_pred.append(pod.predict(y_test))
    avg_predict_time += time.time() - start_time
avg_predict_time /= len(Y_test)

# Format POD predictions for DMD
X_pred = dataset.unstack_simulation_vector(X_pred)
X_test = dataset.unstack_simulation_vector(X_test)

# Compute simulation errors
errors = np.zeros(len(X_test))
for i in range(len(X_test)):
    errors[i] = norm(X_test[i]-X_pred[i]) / norm(X_test[i])

# Get worst-case result
argmax = np.argmax(errors)
x_pred, x_test = X_pred[argmax], X_test[argmax]
timestep_errors = norm(x_test-x_pred, axis=1)/norm(x_test, axis=1)

# Print aggregated POD results
msg = f'===== Summary of {len(errors)} POD Models ====='
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
print(f'Number of Training Snapshots:\t\t{len(X_train)}')
print(f'Number of POD Modes:\t\t\t{pod.n_modes}')
print(f'Average POD Reconstruction Error:\t{np.mean(errors):.3e}')
print(f'Maximum POD Reconstruction Error:\t{np.max(errors):.3e}')
print(f'Minimum POD Reconstruction Error:\t{np.min(errors):.3e}')
print()

# Worst POD result
title = 'Worst Result\n'
if study in [0, 2]:
    title += f'$\\rho$ = {Y[argmax][0]:.3g} $\\frac{{atoms}}{{b-cm}}$'
    title += ', ' if study == 2 else '\n'
if study in [1, 2]:
    title += f'Size = {Y[argmax][study - 1]:.3g} $cm$\n'
title += f'Error = {np.max(errors):.3e}'

plt.figure()
plt.title(title)
plt.xlabel(r'Time [$\mu$s]', fontsize=12)
plt.ylabel('Relative $L^2$ Error', fontsize=12)
plt.semilogy(times, timestep_errors, '-*b')
plt.grid(True)
plt.tight_layout()
plt.show()

# DMD on worst result
dmd = DMD(svd_rank=svd_rank)
dmd.fit(x_pred)
dmd.print_summary()
timestep_errors = dmd.snapshot_errors

plt.figure()
plt.xlabel(f'Time ($\mu$s)', fontsize=12)
plt.ylabel(f'$Relative L^2$ Error', fontsize=12)
plt.semilogy(times, timestep_errors, '-b*')
plt.grid(True)
plt.show()

# base = '/Users/zacharyhardy/Documents/phd/prelim'
# # fname = base + '/figures/worst_dmd_reconstruction.pdf'
# # plt.savefig(fname)

# Interpolation and extrapolation on worst result
mid = len(x_pred) // 2
x = x_pred[:mid + 1:2]
dmd = DMD(svd_rank=svd_rank)
dmd.fit(x)

dmd.dmd_time['tend'] *= 2
dmd.dmd_time['dt'] /= 2
x_dmd = dmd.reconstructed_data

timestep_errors = norm(x_test-x_dmd, axis=1)/norm(x_test, axis=1)

plt.figure()
plt.xlabel('Time ($\mu$s)', fontsize=12)
plt.ylabel('$L^2$ Error', fontsize=12)
plt.semilogy(times[:mid + 1:2], timestep_errors[:mid + 1:2],
             '-*b', markersize=3.0, label='Reconstuction')
plt.semilogy(times[1:mid + 1:2], timestep_errors[1:mid + 1:2],
             '--^r', markersize=3.0, label='Interpolation')
plt.semilogy(times[mid + 1:], timestep_errors[mid + 1:],
             '-.+g', markersize=3.0, label='Extrapolation')

plt.legend()
plt.grid(True)
plt.show()

# # fname = base + '/figures/worst_dmd_interp_extrap.pdf'
# # plt.savefig(fname)
# # plt.show()

# Construct DMD models, compute errors
avg_dmd_time = 0.0
dmd = DMD(svd_rank=svd_rank)
errors = np.zeros(len(X_pred))
for i in range(len(X_pred)):
    # Fit DMD model
    tstart = time.time()
    dmd.fit(X_pred[i])
    avg_dmd_time += time.time() - tstart

    # Compute error
    x_dmd = dmd.reconstructed_data.real
    errors[i] = norm(X_test[i]-x_dmd) / norm(X_test[i])
avg_dmd_time /= len(X_pred)
avg_query_time = avg_predict_time + avg_dmd_time

# Print aggregated DMD results
msg = f'===== Summary of {errors.size} DMD Models ====='
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
print(f'Average DMD Reconstruction Error:\t{np.mean(errors):.3e}')
print(f'Maximum DMD Reconstruction Error:\t{np.max(errors):.3e}')
print(f'Minimum DMD Reconstruction Error:\t{np.min(errors):.3e}')
print()

# Print cost
msg = f'===== Summary of POD-DMD Model Cost ====='
header = '=' * len(msg)
print('\n'.join([header, msg, header]))
print(f'Construction:\t\t\t{construction_time:.3e} s')
print(f'Average Prediction:\t\t{avg_predict_time:.3e} s')
print(f'Average Decomposition:\t\t{avg_dmd_time:.3e} s')
print(f'Average Query Cost:\t\t{avg_query_time:.3e} s')
print()
