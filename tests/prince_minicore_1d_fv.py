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

from prototype_minicore_xs import *

# Create mesh and discretization
zones = [0.0, 16.0, 20.0, 24.0, 56.0, 64.0, 80.0]
n_cells = [80, 20, 20, 160, 40, 80]
material_ids = [0, 1, 2, 0, 3, 0]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs0 = CrossSections()
xs0.read_from_xs_dict(xs_vals)

xs1 = deepcopy(xs0)
xs1.sigma_t_function = sigma_t_1

xs2 = deepcopy(xs0)
xs2.sigma_t_function = sigma_t_2

xs3 = deepcopy(xs0)
xs3.sigma_t_function = sigma_t_3

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
solver.method = "CRANK_NICHOLSON"

solver.max_iterations = 50000
solver.tolerance = 1.0e-12

# Run the problem
solver.initialize()
solver.execute(verbose=1)

out = solver.outputs
sim_times = np.round(out.times, 6)
grid = [p[2] for p in out.grid]

plt.figure()
times = [0.0, 0.6, 1.1, 1.7]
for t in times:
    i = np.argmin(abs(sim_times - t))
    phi = out.flux[i][0]
    plt.plot(grid, phi, label=f"Time = {t:.2g} sec")
plt.legend()
plt.grid(True)

plt.figure()
plt.title("Power Profile")
plt.plot(sim_times, out.power)
plt.grid(True)

plt.show()
