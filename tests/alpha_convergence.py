import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *


abs_path = os.path.dirname(os.path.abspath(__file__))

# Determine the option to use
option = int(sys.argv[1]) if len(sys.argv) > 0 else 0
if option > 1:
    raise AssertionError('Only option 0 and 1 are available.')

# Analytic alpha functions
if option == 0:
    def analytic_alpha(n: List[int]) -> float:
        Bn = (n + 1) * np.pi / r_b
        a = xs.nu_sigma_f[0] - xs.sigma_a[0] - xs.D[0] * Bn ** 2
        return xs.velocity[0] * a
else:
    path = os.path.dirname(os.path.abspath(__file__))
    path += '/three_group_sphere/transient/sphere6cm.obj'
    ex: AnalyticSolution = load(path)

    def analytic_alpha(n: List[int]) -> float:
        return ex.get_mode(n, method='eig').alpha.real

# Mesh parameters
r_b = 10.0 if option == 0 else 6.0
coord_sys = 'cartesian' if option == 0 else 'spherical'

#Create cross sections
material = Material()
xs = CrossSections()

# Read in the correct cross sections
if option == 0:
    xs_vals = {'n_groups': 1, 'sigma_t': [1.0],
               'transfer_matrix': [[0.3]],
               'sigma_f': [0.35],
               'nu': [2.0], 'velocity': [1.0]}
    xs.read_from_xs_dict(xs_vals)
else:
    xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)

material.add_properties(xs)

# Create boundary conditions
boundaries = [ZeroFluxBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

# Initialize solver and attach objects
solver = AlphaEigenvalueSolver()
solver.materials = [material]
solver.boundaries = boundaries

# Initial condition
if option == 0:
    ic_function = [lambda x: 1.0 - (x - 0.5*r_b)**2 / r_b**2]
else:
    ic_function = [lambda x: 1.0 - x**2 / r_b**2,
                   lambda x: 1.0 - x**2 / r_b**2,
                   lambda x: 0.0]

n_cells = [20*2**i for i in range(7)]

errors = []
for i in range(len(n_cells)):
    solver.mesh = create_1d_mesh([0.0, r_b], [n_cells[i]],
                                 coord_sys=coord_sys)

    solver.discretization = FiniteVolume(solver.mesh)

    solver.initialize()
    solver.eigendecomposition(ic_function)

    errors_i = []
    for n in range(solver.n_modes):
        alpha_n = solver.alphas.real[n]
        alpha_e = analytic_alpha(n)
        diff = abs(alpha_e - alpha_n) / abs(alpha_e)
        errors_i.append(diff)
    errors.append(errors_i)

plt.figure()
plt.title(f'{xs.n_groups} Group Problem')
plt.xlabel('Mode Number', fontsize=12)
plt.ylabel('Relative Error', fontsize=12)
for i in range(len(errors)):
    label = f'n = {xs.n_groups * n_cells[i]}'
    plt.semilogy(errors[i][:min(n_cells[i], 200)], label=label)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

from matplotlib.figure import Figure
from matplotlib.axes import Axes
plt.figure()
plt.title('$\\alpha$-Eigenvalue Convergence', fontsize=12)
plt.xlabel('Number of Cells', fontsize=12)
plt.ylabel('Relative Error', fontsize=12)
for n in range(0, 20, 4):
    errors_n = [errors[i][n] for i in range(len(errors))]

    label = f'$\\alpha_{{{n}}}$'
    plt.loglog(n_cells, errors_n, label=label)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
