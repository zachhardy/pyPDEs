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
zones = [0.0, 40.0, 200.0, 240.0]
n_cells = [20, 80, 20]
material_ids = [0, 1, 2]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys='cartesian')
discretization = FiniteVolume(mesh)

# Create materials
materials = []
materials.append(Material('Material 0'))
materials.append(Material('Material 1'))
materials.append(Material('Material 2'))

xs = [CrossSections() for _ in range(len(materials))]
data = [xs_material_0_and_2, xs_material_1, xs_material_0_and_2]
fcts = [sigma_a_ramp_down, None, None]
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
solver.t_final = 4.0
solver.dt = 0.01
solver.method = 'tbdf2'

# Output informations
solver.write_outputs = True
solver.output_directory = \
    os.path.join(abs_path, 'outputs/delayed_supercritical')

# Run the problem
solver.initialize(verbose=1)
solver.execute(verbose=1)
