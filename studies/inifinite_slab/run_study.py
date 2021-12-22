import os
import itertools
import sys

import numpy as np

from copy import deepcopy

from matplotlib import pyplot as plt
from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from modules.neutron_diffusion import *

from xs import *


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f'rm -r {path}/*')


def function(g, x, sigma_a) -> float:
    t = x[0]
    if g == 1 and 0.0 < t <= t_ramp:
        return sigma_a * (1.0 + t/t_ramp*(m - 1.0))
    elif g == 1 and t > t_ramp:
        return m * sigma_a
    else:
        return sigma_a


# Nominal function parameters
m, t_ramp = 1.03, 1.0

# Define current directory
script_path = os.path.dirname(os.path.abspath(__file__))

# Get inputs
case = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if case > 6:
    raise AssertionError('Invalid case to run.')

# Define parameter space
parameters = {}
if case == 0:
    parameters['multiplier'] = np.linspace(1.01, 1.05, 21)
elif case == 1:
    parameters['duration'] = np.linspace(0.75, 1.25, 21)
elif case == 2:
    parameters['interface'] = np.linspace(39.0, 41.0, 21)
elif case == 3:
    parameters['multiplier'] = np.linspace(1.02, 1.04, 6)
    parameters['duration'] = np.linspace(0.9, 1.1, 6)
elif case == 4:
    parameters['multiplier'] = np.linspace(1.02, 1.04, 6)
    parameters['interface'] = np.linspace(38.0, 42.0, 5)
elif case == 5:
    parameters['duration'] = np.linspace(0.9, 1.1, 6)
    parameters['interface'] = np.linspace(38.0, 42.0, 5)
elif case == 6:
    parameters['multiplier'] = np.linspace(1.02, 1.04, 4)
    parameters['duration'] = np.linspace(0.9, 1.1, 4)
    parameters['interface'] = np.linspace(38.0, 42.0, 4)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

study_name = ''
for k, key in enumerate(keys):
    study_name = key if k == 0 else study_name + f'_{key}'

# Create mesh and discretization
zones = [0.0, 40.0, 200.0, 240.0]
n_cells = [20, 80, 20]
material_ids = [0, 1, 2]
mesh = create_1d_mesh(zones, n_cells, material_ids)
discretization = FiniteVolume(mesh)

# Create cross sections and sources
materials = [Material(f'Material {i + 1}') for i in range(3)]
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
solver.t_final = 2.0
solver.dt = 0.02
solver.method = 'tbdf2'

solver.max_iterations = max_iterations
solver.tolerance = tolerance

# Output informations
solver.write_outputs = True

# Define output path
output_path = os.path.join(script_path,
                           f'outputs/subcritical/{study_name}')
setup_directory(output_path)

# Save parameters
param_filepath = os.path.join(output_path, 'params.txt')
np.savetxt(param_filepath, np.array(values), fmt='%.8e')

# Run the reference problem
msg = '===== Running reference ====='
head = '=' * len(msg)
print()
print('\n'.join([head, msg, head]))

simulation_path = os.path.join(output_path, 'reference')
setup_directory(simulation_path)
solver.output_directory = simulation_path
solver.initialize()
solver.execute()

# Run the study
for n, params in enumerate(values):
    msg = f'===== Running simulation {n} ====='
    head = '=' * len(msg)
    print('\n'.join(['', head, msg, head]))
    for p in range(len(params)):
        pname = keys[p].capitalize()
        print(f'{pname:<10}:\t{params[p]:<5.4g}')

    # Setup output path
    simulation_path = os.path.join(output_path, str(n).zfill(3))
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if 'multiplier' in keys and 'duration' not in keys:
        t_ramp = 1.0
        m = params[keys.index('multiplier')]

    if 'duration' in keys and 'multiplier' not in keys:
        m = 1.03
        t_ramp = params[keys.index('duration')]

    if 'multiplier' in keys and 'duration' in keys:
        m = params[keys.index('multiplier')]
        t_ramp = params[keys.index('duration')]

    if 'interface' in keys:
        x_int = params[keys.index('interface')]
        zones = [0.0, x_int, 200.0, 240.0]
        solver.mesh = create_1d_mesh(zones, n_cells, material_ids)
        solver.discretization = FiniteVolume(solver.mesh)
        solver.materials = deepcopy(materials)

    if 'multiplier' in keys or 'duration' in keys:
        solver.materials = deepcopy(materials)
        for material_property in solver.materials[0].properties:
            if isinstance(material_property, CrossSections):
                material_property.sigma_a_function = function

    # Run the problem
    solver.initialize()

    print(solver.k_eff)
    print(solver.material_xs[0].sigma_f)

    solver.execute()
