import os
import sys
import itertools
import time

import numpy as np

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from modules.neutron_diffusion import *
from studies.utils import *

########################################
# Setup parameter study
########################################
path = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 3:
    raise AssertionError(
        f'There must be a command line argument for the '
        f'problem type and parameter study.')

problem = int(sys.argv[1])
if problem > 1:
    raise ValueError('Invalid problem number.')

study = int(sys.argv[2])
if study > 2:
    raise ValueError('Invalid study number.')

# Define all parametric combinations
parameters = {}
if study == 0:
    parameters['density'] = setup_range(0.05, 0.025, 31)
elif study == 1:
    parameters['size'] = setup_range(6.0, 0.025, 31)
else:
    parameters['density'] = setup_range(0.05, 0.0125, 7)
    parameters['size'] = setup_range(6.0, 0.0125, 7)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

# Define the name of the problem
if problem == 0:
    problem_name = 'keigenvalue'
else:
    problem_name = 'ics'

# Define the name of the parameter study
study_name = ''
for k, key in enumerate(keys):
    study_name = key if k == 0 else study_name + f'_{key}'

# Define the path to the output directory
output_path = f'{path}/outputs/{problem_name}/{study_name}'
setup_directory(output_path)

# Save parameter sets
param_filepath = f'{output_path}/params.txt'
np.savetxt(param_filepath, np.array(values), fmt='%.8e')

########################################
# Setup the problem
########################################
mesh = create_1d_mesh([0.0, 6.0], [200], coord_sys='spherical')
discretization = FiniteVolume(mesh)

material = Material()
xs = CrossSections()
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)
material.add_properties(xs)

boundaries = [ReflectiveBoundary(np.zeros(xs.n_groups)),
              ZeroFluxBoundary(np.zeros(xs.n_groups))]

solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = [material]
solver.use_precursors = False

r_b = mesh.vertices[-1].z
ics = [lambda r: 1.0 - r**2/r_b**2,
       lambda r: 1.0 - r**2/r_b**2,
       lambda r: 0.0]
solver.initial_conditions = ics if problem == 1 else None

solver.t_final = 0.1
solver.dt = 2.0e-3
solver.method = 'tbdf2'

solver.adaptivity = True
solver.refine_level = 0.1
solver.coarsen_level = 0.025

solver.write_outputs = True

########################################
# Run the reference problem
########################################
msg = '===== Running reference ====='
head = '=' * len(msg)
print()
print('\n'.join([head, msg, head]))

simulation_path = os.path.join(output_path, 'reference')
setup_directory(simulation_path)
solver.output_directory = simulation_path
solver.initialize()
solver.execute()

########################################
# Run the parameter study
########################################
t_avg = 0.0
for n, params in enumerate(values):

    # Setup output path
    simulation_path = os.path.join(output_path, str(n).zfill(3))
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if 'density' in keys:
        ind = keys.index('density')
        xs.read_from_xs_file('xs/three_grp_us.cxs', density=params[ind])
    if 'size' in keys:
        ind = keys.index('size')
        solver.mesh = create_1d_mesh([0.0, params[ind]], [mesh.n_cells],
                                     coord_sys=mesh.coord_sys)
        solver.discretization = FiniteVolume(mesh)

    # Run the problem
    init_time = time.time()
    solver.initialize()
    init_time = time.time() - init_time

    msg = f'===== Running simulation {n} ====='
    head = '=' * len(msg)
    print('\n'.join(['', head, msg, head]))
    for p in range(len(params)):
        pname = keys[p].capitalize()
        print(f'{pname:<10}:\t{params[p]:<5.3e}')
    print(f"{'k_eff':<10}:\t{solver.k_eff:<8.5f}")

    run_time = time.time()
    solver.execute()
    run_time = time.time() - run_time
    t_avg += (init_time + run_time) / len(values)

print(f'\nAverage simulation time: {t_avg:.3e} s')
