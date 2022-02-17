import os
import sys
import itertools
import time

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_2d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from studies.utils import *
from modules.neutron_diffusion import *
from xs import *


def function(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1:
        if 0.0 <= t <= t_ramp:
            return sigma_a * (1.0 + t/t_ramp*(m - 1.0))
        else:
            return m * sigma_a
    else:
        return sigma_a


########################################
# Setup parameter study
########################################
path = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) != 2:
    raise AssertionError(
        f"There must be a command line argument for the "
        f"parameter set.")

study = int(sys.argv[1])
if study > 6:
    raise ValueError("Invalid study number.")

m_ref, t_ramp_ref = 0.97667, 0.2
m, t_ramp = m_ref, t_ramp_ref

# Define parameter space
parameters = {}
if study == 0:
    parameters['magnitude'] = setup_range(0.98, 0.02, 21)
elif study == 1:
    parameters['duration'] = setup_range(t_ramp_ref, 0.2, 21)
elif study == 2:
    parameters['scatter'] = setup_range(0.01, 0.2, 21)
elif study == 3:
    parameters['magnitude'] = setup_range(0.98, 0.01, 6)
    parameters['duration'] = setup_range(t_ramp_ref, 0.1, 6)
elif study == 4:
    parameters['magnitude'] = setup_range(0.98, 0.01, 6)
    parameters['scatter'] = setup_range(0.01, 0.1, 6)
elif study == 5:
    parameters['duration'] = setup_range(t_ramp_ref, 0.1, 6)
    parameters['scatter'] = setup_range(0.01, 0.1, 6)
else:
    parameters['magnitude'] = setup_range(0.98, 0.01, 4)
    parameters['duration'] = setup_range(t_ramp_ref, 0.1, 4)
    parameters['scatter'] = setup_range(0.01, 0.1, 4)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

# Define the name of the parameter study
study_name = ''
for k, key in enumerate(keys):
    study_name = key if k == 0 else study_name + f'_{key}'

# Define the path to the output directory
output_path = f'{path}/outputs/{study_name}'
setup_directory(output_path)

# Save parameter sets
param_filepath = f'{output_path}/params.txt'
np.savetxt(param_filepath, np.array(values), fmt='%.8e')

########################################
# Setup the problem
########################################
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

solver.adaptivity = True
solver.refine_level = 0.05
solver.coarsen_level = 0.01

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
    if 'magnitude' in keys:
        m = params[keys.index('magnitude')]
    if 'duration' in keys:
        t_ramp = params[keys.index('duration')]

    solver.materials = deepcopy(materials)
    for material_property in solver.materials[0].properties:
        if isinstance(material_property, CrossSections):
            material_property.sigma_a_function = function

    if 'scatter' in keys:
        sig_s = params[keys.index('scatter')]
        for material_property in solver.materials[2].properties:
            if isinstance(material_property, CrossSections):
                material_property._transfer_matrix[0][0][1] = sig_s

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
    print(f"Initial Power:\t{solver.initial_power:.3f}")

    run_time = time.time()
    solver.execute()
    run_time = time.time() - run_time
    t_avg += (init_time + run_time) / len(values)

    print(f"Final Power:\t{solver.power:.3f}")

print(f'\nAverage simulation time: {t_avg:.3e} s')

