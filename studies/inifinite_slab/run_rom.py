import os.path
import sys
import time

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from readers import NeutronicsDatasetReader
from rom.pod import POD
from rom.dmd import DMD


script_path = os.path.dirname(os.path.abspath(__file__))

# Get inputs
case = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if case == 0:
    study_name = 'multiplier'
elif case == 1:
    study_name = 'duration'
elif case == 2:
    study_name = 'interface'
elif case == 3:
    study_name = 'multiplier_duration'
elif case == 4:
    study_name = 'multiplier_interface'
elif case == 5:
    study_name = 'duration_interface'
elif case == 6:
    study_name = 'multiplier_duration_interface'
else:
    raise AssertionError('Invalid case index.')

# Parse the database
path = f'{script_path}/outputs/subcritical/{study_name}'
dataset = NeutronicsDatasetReader(path)
dataset.read_dataset()

# Get the domain information
n_groups = dataset.n_groups
grid = np.array([p.z for p in dataset.nodes])
times = dataset.times

X = dataset.create_dataset_matrix()
Y = dataset.parameters
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

    if not on_bndry:
        interior += [i]
    else:
        bndry += [i]

splits = train_test_split(X[interior], Y[interior], train_size=0.5)
X_train, X_test, Y_train, Y_test = splits
X_train = np.vstack((X_train, X[bndry]))
Y_train = np.vstack((Y_train, Y[bndry]))

# Construct POD model, predict test data
tstart = time.time()
svd_rank = 1.0 - 1.0e-10
pod = POD(svd_rank=svd_rank)
pod.fit(X_train.T, Y_train)
offline_time = time.time() - tstart

pod.plot_singular_values()

tstart = time.time()
X_pred = pod.predict(Y_test, 'linear').T
predict_time = time.time() - tstart

msg = '===== POD Model Summary ====='
header = '=' * len(msg)
print('\n'.join([header, msg, header]))
print(f'# of Modes:\t\t{pod.n_modes}')
print(f'# of Snapshots:\t\t{pod.n_snapshots}')
print(f'Reconstruction Error:\t{pod.reconstruction_error:.3e}')

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
msg = f'===== Summary of {len(errors)} POD Interpolations ====='
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
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

# # Construct DMD models, compute errors
# errors = np.zeros(len(X_pred))
# dmd_time = 0.0
# for i in range(len(X_pred)):
#     tstart = time.time()
#     dmd = DMD(svd_rank=svd_rank, exact=False, opt=True)
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
