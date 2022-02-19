import os
import sys
import time
import itertools

import matplotlib.pyplot as plt
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
        return sigma_a * (1.0 + t/t_ramp*(m - 1.0))
    elif g == 1 and t > t_ramp:
        return m * sigma_a
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
    m_ref, t_ramp_ref = 1.03, 1.0
    t_final, dt = 2.0, 0.04
elif problem == 1:
    m_ref, t_ramp_ref = 0.99, 1.0
    t_final, dt = 2.0, 0.04
else:
    m_ref, t_ramp_ref = 0.95, 0.01
    t_final, dt = 0.02, 4.0e-4

m, t_ramp = m_ref, t_ramp_ref

# Define parameter spaces
parameters = {}
if study == 0:
    parameters['magnitude'] = 1.0 + setup_range(m_ref-1.0, 0.2, 21)[::-1]
elif study == 1:
    parameters['duration'] = setup_range(t_ramp_ref, 0.2, 21)
elif study == 2:
    parameters['interface'] = setup_range(40.0, 0.05, 21)
elif study == 3:
    parameters['magnitude'] = 1.0 + setup_range(m_ref-1.0, 0.2, 6)[::-1]
    parameters['duration'] = setup_range(t_ramp_ref, 0.2, 6)
elif study == 4:
    parameters['magnitude'] = 1.0 + setup_range(m_ref-1.0, 0.1, 6)[::-1]
    parameters['interface'] = setup_range(40.0, 0.0125, 6)
elif study == 5:
    parameters['duration'] = setup_range(t_ramp_ref, 0.2, 6)
    parameters['interface'] = setup_range(40.0, 0.025, 6)
elif study == 6:
    parameters['magnitude'] = 1.0 + setup_range(m_ref-1.0, 0.1, 5)[::-1]
    parameters['duration'] = setup_range(t_ramp_ref, 0.1, 5)
    parameters['interface'] = setup_range(40.0, 0.0125, 5)
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
param_filepath = f"{output_path}/params.txt"
if os.path.isfile(param_filepath):
    # Get old parameters
    all_params = np.loadtxt(param_filepath)
    if all_params.ndim == 1:
        all_params = np.atleast_2d(all_params).T
    all_params = [tuple(np.round(param, 14)) for param in all_params]
    values = [tuple(np.round(value, 14)) for value in values]

    # Figure out new parameters
    new_params = []
    for value in values:
        if value not in all_params:
            all_params.append(value)
            new_params.append(value)

    # Determine starting number
    sim_skip = len(os.listdir(output_path)) - 1
else:
    all_params = new_params = np.round(values, 14)
    sim_skip = 0

print(f"Running {len(new_params)} new simulations...")

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

solver.adaptivity = True
solver.refine_level = 0.05
solver.coarsen_level = 0.01

solver.max_iterations = max_iterations
solver.tolerance = tolerance

# Output informations
solver.write_outputs = True

########################################
# Run the parameter study
########################################
t_avg = 0.0
for n, params in enumerate(new_params):
    with open(param_filepath, 'a') as pfile:
        for par in params:
            pfile.write(f"{par:.14e} ")
        pfile.write("\n")

    # Setup output path
    sim_num = n + sim_skip
    simulation_path = os.path.join(output_path, str(sim_num).zfill(3))
    setup_directory(simulation_path, clear=True)
    solver.output_directory = simulation_path

    # Modify system parameters
    if 'magnitude' in keys and 'duration' not in keys:
        t_ramp = t_ramp_ref
        m = params[keys.index('magnitude')]

    if 'duration' in keys and 'magnitude' not in keys:
        m = m_ref
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
