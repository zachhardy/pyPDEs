import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader

########################################
# Get the path to results
########################################
path = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 4:
    raise AssertionError(
        f'There must be a command line argument to point '
        f'to the problem type, study, and simulation number.')

problem = int(sys.argv[1])
study = int(sys.argv[2])

# Get problem name
if problem == 0:
    problem_name = 'keigenvalue'
elif problem == 1:
    problem_name = 'ics'
else:
    raise ValueError('Invalid problem provided..')

# Get parameter study name
if study == 0:
    study_name = 'density'
elif study == 1:
    study_name = 'size'
elif study == 2:
    study_name = 'density_size'
else:
    raise ValueError('Invalid case provided.')

# Get simulation number
if sys.argv[3] == 'reference':
    sim_name = 'reference'
else:
    sim_name = sys.argv[3].zfill(3)

# Define path
path = f'{path}/outputs/{problem_name}/{study_name}'

# Check path
data_path = f'{path}/{sim_name}'
if not os.path.isdir(data_path):
    raise NotADirectoryError('Invalid path.')

########################################
# Parse the results
########################################
params_path = f'{path}/params.txt'
params = np.loadtxt(params_path)
if params.ndim == 1:
    params = params.reshape(-1, 1)

param_names = data_path.split('/')[-1].split('_')

sim = NeutronicsSimulationReader(data_path)
sim.read_simulation_data()

sim.plot_flux_moments(0, [0, 1, 2], [0.0, 0.01])
plt.show()

from pyROMs.dmd import DMD
X = sim.create_simulation_matrix()
dmd = DMD(svd_rank=1.0-1.0e-12)
dmd.fit(X)
dmd.print_summary()
