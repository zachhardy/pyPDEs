import os
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
    assert len(x) == 4, 'There must be 4 variables in `x` input.'
    t, T, T0, gamma = x[0], x[1], x[2], x[3]

    if g == 0:
        return sigma_a*(1.0 + gamma*(np.sqrt(T) - np.sqrt(T0)))
    elif g == 1:
        if t <= t_ramp:
            return sigma_a*(1.0 + t/t_ramp*delta)
        else:
            return (delta + 1.0)*sigma_a
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

# Nominal parameters
delta_ref, t_ramp_ref, feedback_ref = 0.8787631-1.0, 2.0, 3.034e-3
delta, t_ramp, feedback = delta_ref, t_ramp_ref, feedback_ref

# Define parameter space
parameters = {}
if study == 0:
    parameters['magnitude'] = setup_range(delta_ref, 0.025, 2)
elif study == 1:
    parameters['duration'] = setup_range(t_ramp_ref, 0.05, 2)
elif study == 2:
    parameters['feedback'] = setup_range(feedback_ref, 0.05, 2)
elif study == 3:
    parameters['magnitude'] = setup_range(delta_ref, 0.025, 2)
    parameters['duration'] = setup_range(t_ramp_ref, 0.05, 2)
elif study == 4:
    parameters['magnitude'] =setup_range(delta_ref, 0.025, 6)
    parameters['feedback'] = setup_range(feedback_ref, 0.05, 6)
elif study == 5:
    parameters['duration'] = setup_range(t_ramp_ref, 0.05, 2)
    parameters['feedback'] = setup_range(feedback_ref, 0.05, 2)
else:
    parameters['magnitude'] = setup_range(delta_ref, 0.025, 4)
    parameters['duration'] = setup_range(t_ramp_ref, 0.05, 4)
    parameters['feedback'] = setup_range(feedback_ref, 0.05, 4)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))

# Define the name of the parameter study
study_name = ""
for k, key in enumerate(keys):
    study_name = key if k == 0 else study_name + f"_{key}"

# Define the path to the output directory
output_path = f"{path}/outputs/{study_name}"
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
    all_params = new_params = values
    sim_skip = 0

########################################
# Setup the problem
########################################
# Create mesh, assign material IDs
x_verts = np.linspace(0.0, 165.0, 23)
y_verts = np.linspace(0.0, 165.0, 23)
mesh = create_2d_mesh(x_verts, y_verts, verbose=True)

for cell in mesh.cells:
    c = cell.centroid
    if (0.0 <= c.x <= 15.0 or 75.0 <= c.x <= 105.0) and \
            (0.0 <= c.y <= 15.0 or 75.0 <= c.y <= 105.0):
        cell.material_id = 1
    elif 0.0 <= c.x <= 105.0 and 0.0 <= c.y <= 105.0 and \
            cell.material_id == -1:
        cell.material_id = 0
    elif 0.0 <= c.x <= 105.0 and 105.0 <= c.y <= 135.0:
        cell.material_id = 2
    elif 105.0 <= c.x <= 135.0 and 0.0 <= c.y <= 75.0:
        cell.material_id = 2
    elif 105.0 <= c.x <= 135.0 and 75.0 <= c.y <= 105.0:
        cell.material_id = 3
    elif 105.0 <= c.x <= 120.0 and 105.0 <= c.y <= 120.0:
        cell.material_id = 4
    else:
        cell.material_id = 5

# Create discretizations
discretization = FiniteVolume(mesh)

# Create materials
materials = [Material('Fuel 1 with Rod'),
             Material('Fuel 1 without Rod'),
             Material('Fuel 2 with Rod'),
             Material('Rod Ejection Region'),
             Material('Fuel 2 without Rod'),
             Material('Reflector')]

xs = [CrossSections() for _ in range(len(materials))]
data = [fuel_1_with_rod, fuel_1_without_rod,
        fuel_2_with_rod, fuel_2_with_rod, fuel_2_without_rod,
        reflector]
fcts = [sigma_a_without_rod, sigma_a_without_rod, sigma_a_without_rod,
        sigma_a_with_rod, sigma_a_without_rod, None]
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

# Iterative parameters
solver.tolerance = 1.0e-8
solver.max_iterations = int(1.0e4)

solver.is_nonlinear = True
solver.nonlinear_tolerance = 1.0e-8
solver.nonlinear_max_iterations = 20

# Set precursor options
solver.use_precursors = True
solver.lag_precursors = False

# Set feedback options
solver.feedback_coeff = feedback_ref
solver.energy_per_fission = 3.204e-11
solver.conversion_factor = 3.83e-11

# Initial power
solver.initial_power = 1.0e-6
solver.phi_norm_method = 'average'

# Set time stepping options
solver.t_final = 3.0
solver.dt = 0.01
solver.method = 'tbdf2'

solver.adaptivity = True
solver.refine_level = 0.2
solver.coarsen_level = 0.01

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
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if 'magnitude' in keys:
        delta = params[keys.index('magnitude')]
    if 'duration' in keys:
        t_ramp = params[keys.index('duration')]
    if 'feedback' in key:
        feedback = params[keys.index('feedback')]
        solver.feedback_coeff = feedback

    solver.materials = deepcopy(materials)
    for material_property in solver.materials[3].properties:
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
    print(f"Initial Power:\t{solver.initial_power:.3e}")

    run_time = time.time()
    solver.execute()
    run_time = time.time() - run_time
    t_avg += (init_time + run_time) / len(values)

    print(f"Final Average Power:\t{solver.average_power_density:.3e}\n"
          f"Final Peak Power:\t{solver.peak_power_density:.3e}")

print(f'\nAverage simulation time: {t_avg:.3e} s')
