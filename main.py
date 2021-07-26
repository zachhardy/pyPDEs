import matplotlib.pyplot as plt
import numpy as np

from pyPDEs.mesh.line_mesh import LineMesh
from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.material import CrossSections, MultiGroupSource

from modules.diffusion import TransientSolver
from modules.diffusion import ReflectiveBoudnary, DirichletBoundary

mesh = LineMesh([0.0, 6.0], [50], coord_sys='sphere')
discretization = FiniteVolume(mesh)

xs = CrossSections()
xs.read_from_xs_file('xs/three_grp.cxs', density=0.05)
src = MultiGroupSource(np.zeros(xs.num_groups))

boundaries = [ReflectiveBoudnary(),
              DirichletBoundary(np.zeros(xs.num_groups))]

solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs]
solver.material_src = [src]

solver.use_precursors = True
solver.t_final = 0.1
solver.dt = 2.0e-3

solver.initial_conditions = \
    [lambda r: 1.0 - r ** 2 / mesh.vertices[-1] ** 2,
     lambda r: 1.0 - r ** 2 / mesh.vertices[-1] ** 2,
     lambda r: 0.0 * r]

solver.initialize()
solver.plot_solution(title="Initial Condition")
plt.show()

solver.execute()
solver.plot_solution(title="Final Solution")
plt.show()
