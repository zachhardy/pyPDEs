import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from inf_slab_benchmark_xs import *

# Create mesh and discretization
zones = [0.0, 40.0, 200.0, 240.0]
n_cells = [20, 80, 20]
material_ids = [0, 1, 2]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs0 = CrossSections()
xs0.read_from_xs_dict(xs_material_0_and_2)
xs0.sigma_a_function = sigma_a_ramp_up

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

# Set options
solver.use_precursors = True
solver.lag_precursors = True

# Set time stepping options
solver.t_final = 2.0
solver.dt = 0.01
solver.method = "BACKWARD_EULER"

solver.max_iterations = 50000
solver.tolerance = 1.0e-8

# Run the problem
solver.initialize()
solver.execute(verbose=1)

outputs = solver.outputs
outputs.plot_1d_scalar_flux(times=[0.0, 1.0, 2.0])

plt.show()
