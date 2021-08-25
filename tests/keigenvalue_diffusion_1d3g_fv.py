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

solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs]
solver.material_src = [src]

solver.use_precursors = False
solver.use_groupwise_solver = False

solver.initialize()
solver.execute(verbose=True)
solver.plot_solution(title="Final Solution")
plt.show()
