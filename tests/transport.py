import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.quadratures import *

from modules.neutron_transport import *


# Create mesh and discretization
mesh = create_1d_mesh([0.0, 1.0], [100], coord_sys='cartesian')
discretization = FiniteVolume(mesh)

# Create cross sections and sources
material = Material()

xs = CrossSections()
xs.read_from_xs_file('xs/transport_1g.cxs')

src = IsotropicMultiGroupSource(np.ones(xs.n_groups))

material.add_properties([xs, src])

# Create boundary conditions
boundaries = [VacuumBoundary(), VacuumBoundary()]

# Create angular quadrature
quad = ProductQuadrature(n_polar=2)

# Create solver
solver = SteadyStateSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = [material]
solver.quadrature = quad

solver.initialize()
solver.create_angle_sets()
