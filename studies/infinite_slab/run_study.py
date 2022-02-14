import os
import sys
import time
import itertools

import numpy as np
from copy import deepcopy

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from modules.neutron_diffusion import *
from xs import *
from studies.utils import *


def function(g, x, sigma_a) -> float:
    t = x[0]
    if g == 1 and 0.0 < t <= t_ramp:
        return sigma_a * (1.0 + t/t_ramp*m)
    elif g == 1 and t > t_ramp:
        return (1.0 + m) * sigma_a
    else:
        return sigma_a


########################################
# Setup parameter study
########################################
path = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 3:
    raise AssertionError(
        f'There must be a command line argument for the '
        f'problem description and parameter set.')

problem = int(sys.argv[1])
if problem > 2:
    raise ValueError('Invalid problem number.')

study = int(sys.argv[2])
if study > 6:
    raise ValueError('Invalid study number.')

# Define nominal values for each problem
if problem == 0:
    ramp, duration = 0.03, 1.0
    t_final, dt = 2.0, 0.04
elif problem == 1:
    ramp, duration = -0.01, 1.0
    t_final, dt = 4.0, 0.08
else:
    ramp, duration = -0.05, 0.01
    t_final, dt = 0.02, 4.0e-4

m, t_ramp = ramp, duration

# Define parameter spaces
parameters = {}
if study == 0:
    parameters['magnitude'] = setup_range(ramp, 0.2, 31)
elif study == 1:
    parameters['duration'] = setup_range(duration, 0.2, 31)
elif study == 2:
    parameters['interface'] = setup_range(40.0, 0.05, 31)
elif study == 3:
    parameters['magnitude'] = setup_range(ramp, 0.1, 7)
    parameters['duration'] = setup_range(duration, 0.1, 7)
elif study == 4:
    parameters['magnitude'] = setup_range(ramp, 0.1, 7)
    parameters['interface'] = setup_range(40.0, 0.025, 7)
elif study == 5:
    parameters['duration'] = setup_range(duration, 0.1, 7)
    parameters['interface'] = setup_range(40.0, 0.025, 7)
elif study == 6:
    parameters['magnitude'] = setup_range(ramp, 0.1, 5)
    parameters['duration'] = setup_range(duration, 0.1, 5)
    parameters['interface'] = setup_range(40.0, 0.025, 5)
else:
    raise ValueError(f'Invalid case provided.')

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

# Define the name of the problem
if problem == 0:
    problem_name = 'subcritical'
elif problem == 1:
    problem_name = 'supercritical'
elif problem == 2:
    problem_name = 'prompt_supercritical'
else:
    raise ValueError(f'Invalid problem type provided')

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
# Create mesh and discretization
zones = [0.0, 40.0, 200.0, 240.0]
n_cells = [20, 80, 20]
material_ids = [0, 1, 2]
mesh = create_1d_mesh(zones, n_cells, material_ids)
discretization = FiniteVolume(mesh)

# Create cross sections and sources
materials = [Material('Material 0'),
             Material('Material 1'),
             Material('Material 2')]
xs = [CrossSections() for _ in range(len(materials))]
data = [xs_material_0_and_2, xs_material_1, xs_material_0_and_2]
fcts = [function, None, None]
for i in range(len(materials)):
    xs[i].read_from_xs_dict(data[i])
    xs[i].sigma_a_function = fcts[i]
    materials[i].add_properties(xs[i])

# Create boundary conditions
n_groups = xs[0].n_groups
boundaries = [ZeroFluxBoundary(n_groups),
              ZeroFluxBoundary(n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = deepcopy(mesh)
solver.discretization = deepcopy(discretization)
solver.materials = deepcopy(materials)
solver.boundaries = boundaries

solver.tolerance = tolerance
solver.max_iterations = max_iterations

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = t_final
solver.dt = dt
solver.method = 'tbdf2'

solver.max_iterations = max_iterations
solver.tolerance = tolerance

# Output informations
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
    if 'magnitude' in keys and 'duration' not in keys:
        t_ramp = duration
        m = params[keys.index('magnitude')]

    if 'duration' in keys and 'magnitude' not in keys:
        m = ramp
        t_ramp = params[keys.index('duration')]

    if 'magnitude' in keys and 'duration' in keys:
        m = params[keys.index('magnitude')]
        t_ramp = params[keys.index('duration')]

    if 'interface' in keys:
        x_int = params[keys.index('interface')]
        zones = [0.0, x_int, 200.0, 240.0]
        solver.mesh = create_1d_mesh(zones, n_cells, material_ids)
        solver.discretization = FiniteVolume(solver.mesh)
        solver.materials = deepcopy(materials)

    if 'magnitude' in keys or 'duration' in keys:
        solver.materials = deepcopy(materials)
        for material_property in solver.materials[0].properties:
            if isinstance(material_property, CrossSections):
                material_property.sigma_a_function = function

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
