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
case = 0
with_ics = True
for arg in sys.argv:
    if 'case' in arg:
        case = int(arg.split('=')[1])
    if 'with_ics' in arg:
        with_ics = bool(int(arg.split('=')[1]))

if case == 0:
    study_name = 'density'
elif case == 1:
    study_name = 'size'
elif case == 2:
    study_name = 'density_size'
study_name += '_ics' if with_ics else '_k'

# Parse the database
dataset = NeutronicsDatasetReader(f'{script_path}/outputs/{study_name}')
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

seed = 777
splits = train_test_split(X[interior], Y[interior],
                          train_size=0.5, random_state=seed)
X_train, X_test, Y_train, Y_test = splits
X_train = np.vstack((X_train, X[bndry]))
Y_train = np.vstack((Y_train, Y[bndry]))


# Construct POD model, predict test data
tstart = time.time()
svd_rank = 1.0-1.0e-12
pod = POD(svd_rank=svd_rank)
pod.fit(X_train.T, Y_train)
offline_time = time.time() - tstart

tstart = time.time()
X_pred = pod.predict(Y_test, 'linear').T
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

# Print aggregated POD results
msg = f'===== Summary of {len(errors)} POD Models ====='
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
print(f'Number of Training Snapshots:\t\t{len(X_train)}')
print(f'Average POD Reconstruction Error:\t{np.mean(errors):.3e}')
print(f'Maximum POD Reconstruction Error:\t{np.max(errors):.3e}')
print(f'Minimum POD Reconstruction Error:\t{np.min(errors):.3e}')
print()

# Worst POD result
plt.figure()
plt.title(f'Worst Result\n'
          f'$\\rho$ = {Y[argmax][0]:.3f} $\\frac{{atoms}}{{b-cm}}$\n'
          f'Error = {np.max(errors):.3e}')
plt.xlabel(r'Time [$\mu$s]', fontsize=12)
plt.ylabel('Relative Error [arb. units]', fontsize=12)
plt.semilogy(times, timestep_errors, '-*b')
plt.grid(True)
plt.tight_layout()
plt.show()

# DMD on worst result
dmd = DMD(svd_rank=svd_rank)

dmd.snapshot_time = {'t0': times[0],
                     'tf': times[-1],
                     'dt': np.diff(times)[0]}

dmd.fit(x_pred.T)
x_dmd = dmd.reconstructed_data.T

timestep_errors = []
for t in range(len(x_test)):
    error = norm(x_test[t]-x_dmd[t]) / norm(x_test[t])
    timestep_errors.append(error)

plt.figure()
plt.xlabel(f'Time ($\mu$s)', fontsize=12)
plt.ylabel(f'$L^2$ Error', fontsize=12)
plt.semilogy(times, timestep_errors, '-b*')
plt.grid(True)

base = '/Users/zacharyhardy/Documents/phd/prelim'
fname = base + '/figures/worst_dmd_reconstruction.pdf'
plt.savefig(fname)

# Interpolation and extrapolation on worst result
dmd.snapshot_time['tf'] /= 2.0
dmd.snapshot_time['dt'] *= 2.0

mid = len(x_pred) // 2
x = x_pred[:mid + 1:2]
dmd.fit(x.T)

dmd.plot_timestep_errors()

dmd.dmd_time['tf'] *= 2
dmd.dmd_time['dt'] /= 2

x_dmd = dmd.reconstructed_data.T

timestep_errors = []
for t in range(len(x_test)):
    error = norm(x_test[t]-x_dmd[t]) / norm(x_test[t])
    timestep_errors.append(error)

plt.figure()
plt.xlabel('Time ($\mu$s)', fontsize=12)
plt.ylabel('$L^2$ Error', fontsize=12)
plt.semilogy(times[:mid+1:2], timestep_errors[:mid+1:2],
             '-*b', markersize=3.0, label='Reconstuction')
plt.semilogy(times[1:mid+1:2], timestep_errors[1:mid+1:2],
             '--^r', markersize=3.0, label='Interpolation')
plt.semilogy(times[mid+1:], timestep_errors[mid+1:],
             '-.og', markersize=3.0, label='Extrapolation')
plt.legend()
plt.grid(True)

fname = base + '/figures/worst_dmd_interp_extrap.pdf'
plt.savefig(fname)
plt.show()

# Construct DMD models, compute errors
dmd_time = 0.0
errors = np.zeros(len(X_pred))
for i in range(len(X_pred)):
    tstart = time.time()
    dmd = DMD(svd_rank=svd_rank)
    dmd.fit(X_pred[i].T, verbose=False)
    dmd_time += time.time() - tstart

    x_dmd = dmd.reconstructed_data.real
    errors[i] = norm(X_test[i] - x_dmd.T) / norm(X_test[i])
query_time = predict_time + dmd_time

# Print aggregated DMD results
msg = f'===== Summary of {errors.size} DMD Models ====='
header = '=' * len(msg)
print('\n'.join(['', header, msg, header]))
print(f'Average DMD Reconstruction Error:\t{np.mean(errors):.3e}')
print(f'Maximum DMD Reconstruction Error:\t{np.max(errors):.3e}')
print(f'Minimum DMD Reconstruction Error:\t{np.min(errors):.3e}')
print()

msg = f'===== Summary of POD-DMD Model Cost ====='
header = '=' * len(msg)
print('\n'.join([header, msg, header]))
print(f'Construction:\t\t\t{offline_time:.3e} s')
print(f'Prediction:\t\t\t{predict_time:.3e} s')
print(f'Decomposition:\t\t\t{dmd_time:.3e} s')
print(f'Total query cost:\t\t{query_time:.3e} s')
print()
