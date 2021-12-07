import sys

import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import *
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.quadratures import *

from modules.neutron_transport import *

mode = 0
if len(sys.argv) > 1:
    mode = int(sys.argv[1])
    if mode > 1:
        raise AssertionError('Invalid mode argument.')

# Create mesh and discretization
n_verts = 51 if mode == 1 else 101
x_verts = np.linspace(0, 1.0, n_verts)
mesh = create_2d_mesh(x_verts, x_verts)

if mode == 0:
    for cell in mesh.cells:
        centroid = cell.centroid
        if centroid.x <= 0.1 and centroid.y <= 0.1:
            cell.material_id = 1
        else:
            cell.material_id = 0
elif mode == 1:
    for cell in mesh.cells:
        centroid = cell.centroid
        if 0.45 <= centroid.x <= 0.55 and \
                0.45 <= centroid.y <= 0.55:
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
if mode == 0:
    boundaries = [ReflectiveBoundary(), VacuumBoundary(),
                  ReflectiveBoundary(), VacuumBoundary()]
elif mode == 1:
    boundaries = [VacuumBoundary(), VacuumBoundary(),
                  VacuumBoundary(), VacuumBoundary()]

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
