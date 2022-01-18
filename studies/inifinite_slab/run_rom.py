import os.path
import sys
import time

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from readers import NeutronicsDatasetReader
from pyROMs.pod import POD
from pyROMs.dmd import DMD


########################################
# Get the path to results
########################################
path = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 3:
    raise AssertionError(
        f'There must be a command line argument to point '
        f'to the problem type and study.')

problem = int(sys.argv[1])
if problem > 2:
    raise ValueError('Invalid problem number.')

study = int(sys.argv[2])
if study > 6:
    raise ValueError('Invalid study number')

# Get problem name
if problem == 0:
    problem_name = 'subcritical'
elif problem == 1:
    problem_name = 'supercritical'
else:
    problem_name = 'prompt_supercritical'

# Get parameter study name
if study == 0:
    study_name = 'magnitude'
elif study == 1:
    study_name = 'duration'
elif study == 2:
    study_name = 'interface'
elif study == 3:
    study_name = 'magnitude_duration'
elif study == 4:
    study_name = 'magnitude_interface'
elif study == 5:
    study_name = 'duration_interface'
else:
    study_name = 'magnitude_duration_interface'

# Parse the database
path = f'{path}/outputs/{problem_name}/{study_name}'
dataset = NeutronicsDatasetReader(path)
dataset.read_dataset()

# Get the domain information
n_groups = dataset.n_groups
grid = np.array([p.z for p in dataset.nodes])
times = dataset.times

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

plt.figure()
plt.title(f'Worst Result\n'
          f'Error = {np.max(errors):.3e}')
plt.xlabel('Time (sec)', fontsize=12)
plt.ylabel('Relative Error', fontsize=12)
plt.semilogy(times, timestep_errors, '-*b')
plt.grid(True)
plt.tight_layout()
plt.show()

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
