import os
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from xs import *

abs_path = os.path.dirname(os.path.abspath(__file__))

# Create mesh and discretization
zones = [0.0, 16.0, 20.0, 24.0, 56.0, 64.0, 80.0]
n_cells = [80, 20, 20, 160, 40, 80]
material_ids = [0, 1, 2, 0, 3, 0]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)

# Create materials
materials = []
materials.append(Material("Material 0"))
materials.append(Material("Material 1"))
materials.append(Material("Material 2"))
materials.append(Material("Material 4"))

xs = [CrossSections() for _ in range(len(materials))]
fct = [None, sigma_a_material_1,
        sigma_a_material_2, sigma_a_material_3]

for i in range(len(materials)):
    xs[i].read_from_xs_dict(xs_vals)
    xs[i].sigma_a_function = fct[i]
    materials[i].add_properties(xs[i])

# Create boundary conditions
n_groups = xs[0].n_groups
boundaries = [VacuumBoundary(n_groups),
              VacuumBoundary(n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = materials
solver.boundaries = boundaries

solver.tolerance = tolerance
solver.max_iterations = max_iterations

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 2.0
solver.dt = 0.02
solver.method = "TBDF2"

# Output informations
solver.write_outputs = True
solver.output_directory = \
    os.path.join(abs_path, "outputs/minicore")

# Run the problem
solver.initialize(verbose=1)
solver.execute(verbose=1)
