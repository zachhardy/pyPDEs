import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

def analytic_alpha(n: List[int], g: int = 0) -> float:
    a = xs.nu_sigma_f[g] - xs.nu_sigma_f[g]
    a -= xs.D[g] * ((n + 1) * np.pi / r_b) ** 2
    return xs.velocity[g] * a


abs_path = os.path.dirname(os.path.abspath(__file__))

# Create mesh and discretization
r_b = 1.0
mesh = create_1d_mesh([0.0, r_b], [500], coord_sys='cartesian')
discretization = FiniteVolume(mesh)

# Create cross sections and sources
material = Material()
xs = CrossSections()
xs.read_from_xs_file('xs/fuel_1g.cxs')
material.add_properties(xs)

# Create boundary conditions
boundaries = [ZeroFluxBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

r_b = mesh.vertices[-1].z
initial_conditions = \
    [lambda r: 1.0 - (r - 0.5*r_b)**2 / r_b**2]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = [material]
solver.boundaries = boundaries
solver.initial_conditions = initial_conditions
solver.phi_norm_method = None

# Set time stepping options
solver.t_final = 1.0
solver.dt = 1.0e-3
solver.method = 'tbdf2'

solver.initialize()
aee = solver.solve_alpha_eigenproblem(max_modes=-1, tolerance=1.0e-6)

# Initial  condition
solver.compute_initial_values()
phi_sim = np.copy(solver.phi)
phi_aee = aee.evaluate_expansion(0.0)

# Get n_groups and grid
G = solver.n_groups
grid = [pt.z for pt in discretization.grid]

plt.title('Initial Condition')
for g in range(G):
    plt.plot(grid, phi_sim[g::G], label='Simulation')
    plt.plot(grid, phi_aee[g::G], '--', label=f'$\\alpha$-Solution')
plt.grid(True)
plt.legend()
plt.show()

# Final solution
solver.execute()
phi_sim = solver.phi
phi_aee = aee.evaluate_expansion(solver.t_final)

plt.title(f'Time = {solver.t_final} sec')
for g in range(G):
    plt.plot(grid, phi_sim[g::G], label='Simulation')
    plt.plot(grid, phi_aee[g::G], '--', label=f'$\\alpha$-Solution')
plt.grid(True)
plt.legend()
plt.show()






