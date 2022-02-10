import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *

from modules.neutron_diffusion.boundaries import *
from modules.neutron_diffusion import *

# Create mesh and discretization
# Critical = 6.1612 with \rho = 0.05
mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys='spherical')
discretization = FiniteVolume(mesh)

# Create cross sections and sources
material = Material()
xs = CrossSections()
# Critical = 0.05134325 with r_b = 6.0
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)
src = IsotropicMultiGroupSource(np.ones(xs.n_groups))
material.add_properties([xs, src])

# Create boundary conditions
boundaries = [ReflectiveBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

# Initialize solver and attach objects
solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = [material]

# Set options
solver.use_precursors = False

# Run the problem
solver.initialize()
solver.execute(verbose=True)
solver.plot_solution(title="Final Solution")
plt.show()
