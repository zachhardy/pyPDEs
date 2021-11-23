import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_2d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from xs import *


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f'rm -r {path}/*')


def step(g, x, sigma_a) -> float:
    if g == 1 and x[0] > 0.0:
        return 0.97666 * sigma_a
    else:
        return sigma_a


# Define current directory
script_path = os.path.dirname(os.path.abspath(__file__))

multiplier = np.linspace(0.96, 0.99, 31)

parameters = {}
parameters['multiplier'] = np.round(multiplier, 6)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

# Create mesh, assign material IDs
x_verts = np.linspace(0.0, 80.0, 41)
y_verts = np.linspace(0.0, 80.0, 41)

mesh = create_2d_mesh(x_verts, y_verts, verbose=True)

for cell in mesh.cells:
    c = cell.centroid
    if 24.0 <= c.x <= 56.0 and \
            24.0 <= c.y <= 56.0:
        cell.material_id = 0
    elif 0.0 <= c.x <= 24.0 and \
            24.0 <= c.y <= 56.0:
        cell.material_id = 1
    elif 24.0 <= c.x <= 56.0 and \
            0 <= c.y <= 24.0:
        cell.material_id = 1
    else:
        cell.material_id = 2

# Create discretizations
discretization = FiniteVolume(mesh)

# Create materials
materials = []
materials.append(Material())
materials.append(Material())
materials.append(Material())

xs = [CrossSections() for _ in range(len(materials))]
data = [xs_material_0, xs_material_0, xs_material_1]
fcts = [step, None, None]
for i in range(len(materials)):
    xs[i].read_from_xs_dict(data[i])
    xs[i].sigma_a_function = fcts[i]
    materials[i].add_properties(xs[i])

# Create boundary conditions
n_groups = xs[0].n_groups
boundaries = [ReflectiveBoundary(n_groups),
              VacuumBoundary(n_groups),
              ReflectiveBoundary(n_groups),
              VacuumBoundary(n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = deepcopy(materials)
solver.boundaries = boundaries

solver.tolerance = tolerance
solver.max_iterations = max_iterations

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 0.5
solver.dt = 1.0e-2
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
print(f'Initial Power:\t{solver.power:.3e} W')
solver.execute()
print(f'Final Power:\t{solver.power:.3e} W')

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
    if 'multiplier' in keys:
        ind = keys.index('multiplier')
        def function(g, x, sigma_a) -> float:
            if g == 1 and x[0] > 0.0:
                return params[ind] * sigma_a
            else:
                return sigma_a

        solver.materials = deepcopy(materials)
        for material_property in solver.materials[0].properties:
            if isinstance(material_property, CrossSections):
                material_property.sigma_a_function = function

    # Run the problem
    solver.initialize()
    print(f'Initial Power:\t{solver.power:.3e} W')
    solver.execute()
    print(f'Final Power:\t{solver.power:.3e} W')
