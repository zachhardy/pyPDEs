import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from modules.neutron_diffusion.boundaries import *
from modules.neutron_diffusion import SteadyStateSolver

# Create mesh and discretization
mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys='spherical')
discretization = FiniteVolume(mesh)

# Create cross sections and sources
material = Material()
xs = CrossSections()
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)
src = IsotropicMultiGroupSource(np.ones(xs.n_groups))
material.add_properties([xs, src])

# Create boundary conditions
boundaries = [ReflectiveBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

# Initialize solver and attach objects
solver = SteadyStateSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = [material]

# Run the problem
solver.initialize()
solver.execute()
solver.plot_solution(title='Final Solution')
plt.show()
