import time

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

# Create mesh and discretization
zones = [0.0, 80.0, 100.0, 120.0, 280.0, 320.0, 400.0]
n_cells = [80, 20, 20, 160, 40, 80]
material_ids = [0, 1, 2, 0, 3, 0]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs0 = CrossSections()
xs0.read_from_xs_dict(xs_vals)

xs1 = deepcopy(xs0)
xs1.sigma_a_function = sigma_a_material_1

xs2 = deepcopy(xs0)
xs2.sigma_a_function = sigma_a_material_2

xs3 = deepcopy(xs0)
xs3.sigma_a_function = sigma_a_material_3

# Create boundary conditions
boundaries = [VacuumBoundary(xs0.n_groups),
              VacuumBoundary(xs0.n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs0, xs1, xs2, xs3]

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 2.0
solver.dt = 0.01
solver.method = "TBDF2"

solver.max_iterations = 50000
solver.tolerance = 1.0e-12

# Run the problem
solver.initialize()
solver.execute(verbose=1)

outputs = solver.outputs
outputs.plot_1d_scalar_flux(times=[0.0, 0.6, 1.1, 2.0])
outputs.plot_system_power(normalize=True)
plt.show()