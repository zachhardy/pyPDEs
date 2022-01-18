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
if problem > 2:
    raise ValueError('Invalid problem number.')

study = int(sys.argv[2])
if study > 6:
    raise ValueError('Invalid study number.')

# Get problem name
if problem == 0:
    problem_name = 'subcritical'
    times = [0.0, 1.0, 2.0]
elif problem == 1:
    problem_name = 'supercritical'
    times = [0.0, 1.0, 4.0]
else:
    problem_name = 'prompt_supercritical'
    times = [0.0, 0.01, 0.02]

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

sim.plot_flux_moments(0, [0, 1], times, grouping='time')
plt.tight_layout()
plt.show()
