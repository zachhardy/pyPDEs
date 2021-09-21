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
x_verts = np.linspace(0.0, 165.0, 56)
y_verts = np.linspace(0.0, 165.0, 56)
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
    elif 105.0 <= c.x <= 135.0 and 0.0 <= c.y <= 105.0:
        cell.material_id = 2
    elif 105.0 <= c.x <= 120.0 and 105.0 <= c.y <= 120.0:
        cell.material_id = 3
    else:
        cell.material_id = 4

mesh.plot_material_ids()

# Create discretizations
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs0 = CrossSections()
xs0.read_from_xs_dict(fuel_1_with_rod)
xs0.sigma_t_function = sigma_a_generic

xs1 = CrossSections()
xs1.read_from_xs_dict(fuel_1_without_rod)
xs1.sigma_a_function = sigma_a_generic

xs2 = CrossSections()
xs2.read_from_xs_dict(fuel_2_with_rod)
xs2.sigma_a_function = sigma_a_generic

xs3 = CrossSections()
xs3.read_from_xs_dict(fuel_2_without_rod)
xs3.sigma_a_function = sigma_a_generic

xs4 = CrossSections()
xs4.read_from_xs_dict(reflector)
xs4.sigma_a_function = sigma_a_generic

# Create boundary conditions
boundaries = [ReflectiveBoundary(xs0.n_groups),
              ZeroFluxBoundary(xs0.n_groups),
              ZeroFluxBoundary(xs0.n_groups),
              ReflectiveBoundary(xs0.n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs0, xs1, xs2, xs3, xs4]

# Iterative parameters
solver.max_iterations = 50000
solver.tolerance = 1.0e-8

# Set precursor options
solver.use_precursors = True
solver.lag_precursors = False

# Set feedback options
solver.use_feedback = True
solver.feedback_coeff = 2.034e-3
solver.feedback_groups = [0]

# Initial power
solver.power = 1.0e-6

# Set time stepping options
solver.t_final = 3.0
solver.dt = 2.5e-3
solver.method = "CRANK_NICHOLSON"

# Run the problem
solver.initialize()
solver.plot_flux()
plt.show()
sys.exit()

solver.execute(verbose=1)

solver.outputs.plot_flux(0, 1.44, f"Group 0 at t = 1.44 sec")
solver.outputs.plot_power(logscale=True)

plt.show()
