import sys

import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import *
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.quadratures import *

from modules.neutron_transport import *


# Create mesh and discretization
x_verts = np.linspace(0, 1.0, 26)
mesh = create_2d_mesh(x_verts, x_verts)

for cell in mesh.cells:
    centroid = cell.centroid
    if centroid.x <= 0.1 and centroid.y <= 0.1:
        cell.material_id = 1
    else:
        cell.material_id = 0

discretization = FiniteVolume(mesh)

# Create cross sections and sources
materials = []
materials.append(Material('Absorber'))
materials.append(Material('Source'))

xs_vals = {'n_groups': 1, 'sigma_t': [1.0],
           'transfer_matrix': [[[0.1]]]}
xs = CrossSections()
xs.read_from_xs_dict(xs_vals)

materials[0].add_properties(xs)
materials[1].add_properties(xs)

src = IsotropicMultiGroupSource(np.ones(1))
materials[1].add_properties(src)

# Create boundary conditions
boundaries = [ReflectiveBoundary(), VacuumBoundary(),
              ReflectiveBoundary(), VacuumBoundary()]

# Create angular quadrature
quad = ProductQuadrature(4, 2, quadrature_type='glc')

# Create solver
solver = SteadyStateSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = materials
solver.quadrature = quad

solver.scattering_order = 0
solver.max_source_iterations = 100
solver.source_iteration_tolerance = 1.0e-6

solver.initialize()
solver.execute()

solver.plot_flux_moment(ell=0, m=0, group_nums=0)
plt.show()
