import sys

import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_2d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *
from time import time

from modules.diffusion import *


print(f"\n===== Creating the mesh")
x_verts = np.linspace(0.0, 10.0, 41)
y_verts = np.linspace(0.0, 10.0, 41)
mesh = create_2d_mesh(x_verts, y_verts)
print(f"===== Mesh created")

print(f"\n===== Assigning material IDs")
fuel_dim = 7.5
for cell in mesh.cells:
    vids = cell.vertex_ids
    c = cell.centroid
    if c.x <= fuel_dim and c.y <= fuel_dim:
        cell.material_id = 0
    else:
        cell.material_id = 1
print(f"===== Material IDs assigned")

print(f"\n===== Creating discretization")
discretization = FiniteVolume(mesh)
print(f"===== Discretization created.")

print(f"\n====== Creating materials")
xs_fuel = CrossSections()
xs_fuel.read_from_xs_file("xs/fuel_1g.cxs")

xs_refl = CrossSections()
xs_refl.read_from_xs_file("xs/reflector_1g.cxs")

src_fuel = MultiGroupSource(np.ones(xs_fuel.n_groups))
src_refl = MultiGroupSource(np.zeros(xs_refl.n_groups))
print(f"===== Materialis created")

print(f"\n===== Creating boundaries")
boundaries = [ReflectiveBoundary()] * 2
boundaries.extend([VacuumBoundary()] * 2)
print(f"\n===== Boundaries created")

solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs_fuel, xs_refl]
solver.material_src = [src_fuel, src_refl]

solver.max_iterations = 2500
solver.tolerance = 1.0e-6

solver.initialize()
solver.execute()
solver.plot_solution()
plt.show()
