import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import *
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.quadratures import *

from modules.neutron_transport import *


# Create mesh and discretization
mesh = create_1d_mesh([0.0, 1.0], [10])
discretization = FiniteVolume(mesh)

# Create cross sections and sources
material = Material()

xs = CrossSections()
xs.read_from_xs_file('xs/transport_1g.cxs')
material.add_properties(xs)

src = IsotropicMultiGroupSource(np.ones(xs.n_groups))
material.add_properties(src)

# Create boundary conditions
boundaries = [VacuumBoundary(), ReflectiveBoundary()]

# Create angular quadrature
quad = ProductQuadrature(4, quadrature_type='gl')

# Create solver
solver = SteadyStateSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = [material]
solver.quadrature = quad

solver.scattering_order = 0
solver.max_iterations = 100
solver.tolerance = 1.0e-6

solver.initialize()
solver.execute()
solver.plot_flux_moment(0, 0, 0)


