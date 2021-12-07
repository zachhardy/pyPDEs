import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import *
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.quadratures import *

from modules.neutron_transport import *


# Create mesh and discretization
mesh = create_1d_mesh([0.0, 0.2, 0.9, 1.0], [20, 70, 10], [1, 0, 1])
discretization = FiniteVolume(mesh)

# Create cross sections and sources
materials = []
materials.append(Material('Absorber'))
materials.append(Material('Reflector'))

absorber_xs = {'n_groups': 1, 'sigma_t': [10.0],
               'transfer_matrix': [[[1.0]]]}
reflector_xs = {'n_groups': 1, 'sigma_t': [5.0],
                'transfer_matrix': [[[4.9]]]}

xs_absorber = CrossSections()
xs_reflector = CrossSections()

xs_absorber.read_from_xs_dict(absorber_xs)
xs_reflector.read_from_xs_dict(reflector_xs)

materials[0].add_properties(xs_absorber)
materials[1].add_properties(xs_reflector)

src = IsotropicMultiGroupSource(np.ones(1))
materials[0].add_properties(src)

# Create boundary conditions
boundaries = [ReflectiveBoundary(), VacuumBoundary()]

# Create angular quadrature
quad = ProductQuadrature(4, quadrature_type='gl')

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
