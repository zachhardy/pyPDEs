import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from xs import *


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f"rm -r {path}/*")


def step_up(g, x, sigma_a) -> float:
    if g == 1 and x[0] > 0.0:
        return 1.03 * sigma_a
    else:
        return sigma_a


script_path = os.path.dirname(os.path.abspath(__file__))

# Define current directory
script_path = os.path.dirname(os.path.abspath(__file__))

# Define paramter space
multiplier = np.linspace(1.01, 1.04, 31)

parameters = {}
parameters["multiplier"] = np.round(multiplier, 6)

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))


# Create mesh and discretization
zones = [0.0, 40.0, 200.0, 240.0]
n_cells = [20, 80, 20]
material_ids = [0, 1, 2]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs0 = CrossSections()
xs0.read_from_xs_dict(xs_material_0_and_2)
xs0.sigma_a_function = step_up

xs1 = CrossSections()
xs1.read_from_xs_dict(xs_material_1)

xs2 = CrossSections()
xs2.read_from_xs_dict(xs_material_0_and_2)

# Create boundary conditions
boundaries = [ZeroFluxBoundary(xs0.n_groups),
              ZeroFluxBoundary(xs0.n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs0, xs1, xs2]

solver.tolerance = tolerance
solver.max_iterations = max_iterations

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 2.0
solver.dt = 0.04
solver.method = "TBDF2"

solver.max_iterations = max_iterations
solver.tolerance = tolerance

# Output informations
solver.write_outputs = True

# Define output path
output_path = os.path.join(script_path, "outputs/subcritical")
setup_directory(output_path)

# Save parameters
param_filepath = os.path.join(output_path, "params.txt")
np.savetxt(param_filepath, np.array(values), fmt="%.8e")

# Run the reference problem
msg = "===== Running reference ====="
head = "=" * len(msg)
print()
print("\n".join([head, msg, head]))

simulation_path = os.path.join(output_path, "reference")
setup_directory(simulation_path)
solver.output_directory = simulation_path
solver.initialize()
solver.execute()

# Run the study
for n, params in enumerate(values):
    msg = f"===== Running simulation {n} ====="
    head = "=" * len(msg)
    print("\n".join(["", head, msg, head]))
    for p in range(len(params)):
        pname = keys[p].capitalize()
        print(f"{pname:<10}:\t{params[p]:<5}")

    # Setup output path
    simulation_path = os.path.join(output_path, str(n).zfill(3))
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if "multiplier" in keys:
        ind = keys.index("multiplier")
        def function(g, x, sigma_a) -> float:
            if g == 1 and x[0] > 0.0:
                return params[ind] * sigma_a
            else:
                return sigma_a
        solver.material_xs[0].sigma_a_function = function

    # Run the problem
    solver.initialize()
    solver.execute()
