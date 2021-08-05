import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.diffusion import *

mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys="SPHERICAL")
discretization = FiniteVolume(mesh)

xs = CrossSections()
xs.read_from_xs_file('xs/three_grp.cxs', density=0.05)

src = MultiGroupSource(np.zeros(xs.n_groups))

boundaries = []
boundaries.extend([ReflectiveBoundary()] * xs.n_groups)
boundaries.extend([ZeroFluxBoundary()] * xs.n_groups)

solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs]
solver.material_src = [src]

solver.use_precursors = True
solver.lag_precursors = True
solver.t_final = 0.1
solver.dt = 1.0e-3
solver.stepping_method = "TBDF2"

solver.initial_conditions = \
    [lambda r: 1.0 - r ** 2 / mesh.vertices[-1] ** 2,
     lambda r: 1.0 - r ** 2 / mesh.vertices[-1] ** 2,
     lambda r: 0.0 * r]

solver.initialize()

# solver.plot_solution(title="Initial Condition")
# plt.show()

solver.execute()
solver.plot_solution(title="Final Solution")
plt.show()
