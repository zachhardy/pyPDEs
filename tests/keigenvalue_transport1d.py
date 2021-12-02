import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import *
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.quadratures import *

from modules.neutron_transport import *


# Create mesh and discretization
mesh = create_1d_mesh([0.0, 10.0], [100])
discretization = FiniteVolume(mesh)

# Create cross sections and sources
materials = []
materials.append(Material('Fuel'))

xs_fuel = CrossSections()
xs_fuel.read_from_xs_file('xs/fuel_1g.cxs')
materials[0].add_properties(xs_fuel)

# Create boundary conditions
boundaries = [VacuumBoundary(), VacuumBoundary()]

# Create angular quadrature
quad = ProductQuadrature(4, quadrature_type='gl')

# Create solver
solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = materials
solver.quadrature = quad

solver.scattering_order = 0

solver.max_source_iterations = 100
solver.source_iteration_tolerance = 1.0e-6

solver.max_power_iterations = 1000
solver.power_iteration_tolerance = 1.0e-8

solver.initialize()
solver.execute()

solver.plot_flux_moment(ell=0, m=0, group_num=0)
plt.show()
