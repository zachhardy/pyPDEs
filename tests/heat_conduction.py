import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities.boundaries import *

from modules.heat_conduction import HeatConductionSolver

mesh = create_1d_mesh([0.0, 0.45], [50], coord_sys="CARTESIAN")
discretization = PiecewiseContinuous(mesh, degree=1)
boundaries = [NeumannBoundary(0.0), DirichletBoundary(0.0)]

k = [lambda temp: 1.5 + (2510.0 / (215.0 + temp))]
q = [3.0e4]

solver = HeatConductionSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.k = k
solver.q = q

methods = ["PICARD", "NEWTON_DIRECT",
           "NEWTON_GMRES", "NEWTON_JFNK"]
u = []

for method in methods:
    msg = "=========="
    msg += f" Starting {method} Execution "
    msg += "=========="
    print("\n".join(["=" * len(msg), msg, "=" * len(msg)]))

    solver.initialize()
    solver.nonlinear_method = method
    solver.execute(verbose=True)
    u.append(solver.u)


labels = ["Picard", "Newton Direct",
          "Newton GMRES", "Newton JFNK"]
lines = ["-ob", "--.r", "-.g", "^y"]
x = solver.discretization.grid

for i in range(len(u)):
    plt.plot(x, u[i], lines[i], label=labels[i])

plt.title("Solutions")
plt.xlabel("Location")
plt.ylabel(r"T(r)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.show()
