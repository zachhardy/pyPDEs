import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

# Create mesh and discretization
mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys="SPHERICAL")
discretization = PiecewiseContinuous(mesh, degree=2)

# Create materials
materials = [Material()]
xs = CrossSections()
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)
materials[0].add_properties([xs])


# Create boundary conditions
boundaries = [ReflectiveBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

# Initialize solver and attach objects
solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = materials
solver.boundaries = boundaries

# Set options
solver.use_precursors = True

# Run the problem
solver.initialize()
solver.execute(verbose=True)
solver.plot_solution(title="Final Solution")
plt.show()
