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
case = int(sys.argv[2])


# Get problem name
if problem == 0:
    problem_name = 'keigenvalue'
elif problem == 1:
    problem_name = 'ics'
else:
    raise ValueError('Invalid problem provided..')

# Get parameter study name
if case == 0:
    case_name = 'density'
elif case == 1:
    case_name = 'size'
elif case == 2:
    case_name = 'density_size'
else:
    raise ValueError('Invalid case provided.')

# Get simulation number
if sys.argv[3] == 'reference':
    sim_name = 'reference'
else:
    sim_name = sys.argv[3].zfill(3)

# Define path
path = f'{path}/outputs/{problem_name}/{case_name}/{sim_name}'

# Check path
if not os.path.isdir(path):
    raise NotADirectoryError('Invalid path.')

########################################
# Parse the results
########################################
sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, [0, 1, 2], [0.0, 0.01])
plt.show()

from pyROMs.dmd import DMD
X = sim.create_simulation_matrix()
dmd = DMD(svd_rank=1.0-1.0e-12)
dmd.fit(X)
dmd.print_summary()
