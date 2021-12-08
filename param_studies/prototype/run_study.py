import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from xs import xs_vals, tolerance, max_iterations


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f'rm -r {path}/*')


def sigma_a_function(g, x, sigma_a) -> float:
    return 1.1 if x[0] == 0.0 else 1.09

# Define current directory
script_path = os.path.dirname(os.path.abspath(__file__))

# Define paramter space
parameters = {}
parameters['sigma_a'] = np.linspace(1.09, 1.1, 21)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))


# Create mesh and discretization
zones = [0.0, 16.0, 20.0, 24.0, 56.0, 64.0, 80.0]
n_cells = [80, 20, 20, 160, 40, 80]
material_ids = [0, 1, 2, 0, 3, 0]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys='cartesian')
discretization = FiniteVolume(mesh)

# Create cross sections and sources
materials = []
materials.append(Material('Material 0'))
materials.append(Material('Material 1'))
materials.append(Material('Material 2'))
materials.append(Material('Material 4'))

xs = [CrossSections() for _ in range(len(materials))]
fct = [None, sigma_a_function, None,
       sigma_a_function]
for i in range(len(materials)):
    xs[i].read_from_xs_dict(xs_vals)
    xs[i].sigma_a_function = fct[i]
    materials[i].add_properties(xs[i])


# Create boundary conditions
n_groups = xs_vals['n_groups']
boundaries = [VacuumBoundary(n_groups),
              VacuumBoundary(n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = materials

solver.tolerance = tolerance
solver.max_iterations = max_iterations

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 2.0
solver.dt = 0.02
solver.method = 'tbdf2'

# Output informations
solver.write_outputs = True

# Define output path
output_path = os.path.join(script_path, 'outputs')
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
        print(f'{pname:<10}:\t{params[p]:<5}')

    # Setup output path
    simulation_path = os.path.join(output_path, str(n).zfill(3))
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if 'sigma_a' in keys:
        ind = keys.index('sigma_a')
        def function(g, x, sigma_a) -> float:
            return 1.1 if x[0] == 0.0 else params[ind]
        solver.material_xs[1].sigma_a_function = function
        solver.material_xs[3].sigma_a_function = function

    # Run the problem
    solver.initialize()
    solver.execute()
