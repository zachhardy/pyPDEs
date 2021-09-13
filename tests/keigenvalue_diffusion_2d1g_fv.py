import sys

import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_2d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *
from time import time

from modules.neutron_diffusion import *

# Create mesh, assign material IDs
x_verts = np.linspace(0.0, 10.0, 41)
y_verts = np.linspace(0.0, 10.0, 41)
mesh = create_2d_mesh(x_verts, y_verts, verbose=True)

fuel_dim = 9.0
for cell in mesh.cells:
    vids = cell.vertex_ids
    c = cell.centroid
    if c.x <= fuel_dim and c.y <= fuel_dim:
        cell.material_id = 0
    else:
        cell.material_id = 1

# Create discretizations
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs_fuel = CrossSections()
xs_fuel.read_from_xs_file("xs/fuel_1g.cxs")
src_fuel = MultiGroupSource(np.ones(xs_fuel.n_groups))

xs_refl = CrossSections()
xs_refl.read_from_xs_file("xs/reflector_1g.cxs")
src_refl = MultiGroupSource(np.zeros(xs_refl.n_groups))

# Create boundary conditions
boundaries = [ReflectiveBoundary(), VacuumBoundary(),
              VacuumBoundary(), ReflectiveBoundary()]

# Initialize solver and attach objects
solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs_fuel, xs_refl]
solver.material_src = [src_fuel, src_refl]

# Set options
solver.max_iterations = 2500
solver.tolerance = 1.0e-8

# Run the problem
solver.initialize()
solver.execute(verbose=1)
solver.plot_solution()
plt.show()
