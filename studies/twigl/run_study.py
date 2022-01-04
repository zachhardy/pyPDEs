import os
import sys
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


def function(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1:
        if 0.0 <= t <= t_ramp:
            return sigma_a * (1.0 + t/t_ramp*(m - 1.0))
        else:
            return m * sigma_a
    else:
        return sigma_a


# Nominal parameters
m = 0.97667
t_ramp = 0.2

# Define current directory
script_path = os.path.dirname(os.path.abspath(__file__))

# Get inputs
case = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if case > 2:
    raise AssertionError('Invalid case to run.')

# Define parameter space
parameters = {}
if case == 0:
    parameters['multiplier'] = np.linspace(0.97, 0.98, 21)
elif case == 1:
    parameters['duration'] = np.linspace(0.15, 0.25, 21)
elif case == 2:
    parameters['multiplier'] = np.linspace(0.9725, 0.9775, 6)
    parameters['duration'] = np.linspace(0.175, 0.2225, 6)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

study_name = ''
for k, key in enumerate(keys):
    study_name = key if k == 0 else study_name + f'_{key}'

# Create mesh, assign material IDs
x_verts = np.linspace(0.0, 80.0, 21)
y_verts = np.linspace(0.0, 80.0, 21)

mesh = create_2d_mesh(x_verts, y_verts, verbose=True)

for cell in mesh.cells:
    c = cell.centroid
    if 24.0 <= c.x <= 56.0 and 24.0 <= c.y <= 56.0:
        cell.material_id = 0
    elif 0.0 <= c.x <= 24.0 and 24.0 <= c.y <= 56.0:
        cell.material_id = 1
    elif 24.0 <= c.x <= 56.0 and 0 <= c.y <= 24.0:
        cell.material_id = 1
    else:
        cell.material_id = 2

# Create discretizations
discretization = FiniteVolume(mesh)

# Create materials
materials = [Material('Material 1'),
             Material('Material 2'),
             Material('Material 3')]

xs = [CrossSections() for _ in range(len(materials))]
data = [xs_material_0, xs_material_0, xs_material_1]
fcts = [function, None, None]
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
output_path = os.path.join(script_path, f'outputs/{study_name}')
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
        print(f'{pname:<10}:\t{params[p]:<5g}')

    # Setup output path
    simulation_path = os.path.join(output_path, str(n).zfill(3))
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if 'multiplier' in keys and 'duration' not in keys:
        t_ramp = 0.2
        m = params[keys.index('multiplier')]
    if 'duration' in keys and 'multiplier' not in keys:
        m = 0.97667
        t_ramp = params[keys.index('duration')]
    if 'multiplier' in keys and 'duration' in keys:
        m = params[keys.index('multiplier')]
        t_ramp = params[keys.index('duration')]

    solver.materials = deepcopy(materials)
    for material_property in solver.materials[0].properties:
        if isinstance(material_property, CrossSections):
            material_property.sigma_a_function = function

    # Run the problem
    solver.initialize()
    print(f'Initial Power:\t{solver.power:.3e} W')
    solver.execute()
    print(f'Final Power:\t{solver.power:.3e} W')
