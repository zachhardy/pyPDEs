import sys

import numpy as np
import matplotlib.pyplot as plt

from time import time
from copy import deepcopy

from pyPDEs.mesh import create_2d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from xs import *

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

# Create cross sections and sources
xs0 = CrossSections()
xs0.read_from_xs_dict(xs_material_0)
xs0.sigma_a_function = sigma_a_step

xs1 = CrossSections()
xs1.read_from_xs_dict(xs_material_0)

xs2 = CrossSections()
xs2.read_from_xs_dict(xs_material_1)

# Create boundary conditions
boundaries = [ReflectiveBoundary(xs0.n_groups),
              VacuumBoundary(xs0.n_groups),
              VacuumBoundary(xs0.n_groups),
              ReflectiveBoundary(xs0.n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs0, xs1, xs2]

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 0.5
solver.dt = 1.0e-2
solver.method = "TBDF2"


solver.max_iterations = 50000
solver.tolerance = 1.0e-8

# Run the problem
solver.initialize()
solver.execute(verbose=1)

outputs = solver.outputs
outputs.plot_2d_scalar_flux(times=[0.0, 0.2, 0.5])
outputs.plot_system_power(normalize=True)

plt.show()