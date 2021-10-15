import os
import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, IsotropicMultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

abs_path = os.path.dirname(os.path.abspath(__file__))

# Create mesh and discretization
mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys="SPHERICAL")
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs = CrossSections()
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)
src = IsotropicMultiGroupSource(np.zeros(xs.n_groups))

# Create boundary conditions
boundaries = [ReflectiveBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs]
solver.material_src = [src]

# Create and attach initial conditions
rf = mesh.vertices[-1].z
solver.initial_conditions = \
    [lambda r: 1.0 - r ** 2 / rf ** 2,
     lambda r: 1.0 - r ** 2 / rf ** 2,
     lambda r: 0.0 * r]

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 0.1
solver.dt = 2.0e-3
solver.method = "TBDF2"

# Output informations
solver.write_outputs = True
solver.output_directory = \
    os.path.join(abs_path, "outputs/fv")

# Run the problem
solver.initialize(verbose=1)
solver.execute(verbose=1)
