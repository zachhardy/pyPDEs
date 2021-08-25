import sys

import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.diffusion import *

# Create mesh and discretization
mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys="SPHERICAL")
discretization = PiecewiseContinuous(mesh)

# Create cross sections and sources
xs = CrossSections()
xs.read_from_xs_file('xs/three_grp.cxs', density=0.05)
src = MultiGroupSource(np.zeros(xs.n_groups))

# Create boundary conditions
boundaries = []
boundaries.extend([ReflectiveBoundary()] * xs.n_groups)
boundaries.extend([ZeroFluxBoundary()] * xs.n_groups)

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs]
solver.material_src = [src]

# Create and attach initial conditions
solver.initial_conditions = \
    [lambda r: 1.0 - r ** 2 / mesh.vertices[-1].z ** 2,
     lambda r: 1.0 - r ** 2 / mesh.vertices[-1].z ** 2,
     lambda r: 0.0 * r]

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 0.1
solver.dt = 2.0e-3
solver.stepping_method = "TBDF2"

# Run the problem
solver.initialize()
solver.execute(verbose=True)
solver.plot_solution(title="Final Solution")
plt.show()
