import time

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *


def sigma_t_1(t: float, sigt_i: float) -> float:
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.095 - 1.1)
    else:
        return 1.095

def sigma_t_2(t: float, sigt_i: float) -> float:
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.09 - 1.1)
    elif 0.6 < t <= 1.0:
        return 1.09
    elif 1.0 < t <= 1.7:
        f = (t - 1.0) / 0.7
        return 1.09 + f*(1.1 - 1.09)
    else:
        return 1.1

def sigma_t_3(t: float, sigt_i: float) -> float:
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.105 - 1.1)
    else:
        return 1.105


# Create mesh and discretization
zones = [0.0, 80.0, 100.0, 120.0, 280.0, 320.0, 400.0]
n_cells = [80, 20, 20, 160, 40, 80]
material_ids = [0, 1, 2, 0, 3, 0]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)

# Create cross sections and sources
xs_vals = {"n_groups": 1, "n_precursors": 1,
           "D": [1.0], "sigma_t": [1.1], "sigma_f": [1.1],
           "transfer_matrix": [[0.0]],
           "velocity": [1000.0],
           "nu_prompt": [0.994], "nu_delayed": [0.006],
           "precursor_lambda": [0.1], "precursor_yield": [1.0]}


xs0 = CrossSections()
xs0.read_from_xs_dict(xs_vals)

xs1 = deepcopy(xs0)
xs1.sigma_t_function = sigma_t_1

xs2 = deepcopy(xs0)
xs2.sigma_t_function = sigma_t_2

xs3 = deepcopy(xs0)
xs3.sigma_t_function = sigma_t_3

# Create boundary conditions
boundaries = [VacuumBoundary(xs0.n_groups),
              VacuumBoundary(xs0.n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.material_xs = [xs0, xs1, xs2, xs3]

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 2.0
solver.dt = 0.01
solver.method = "TBDF2"
solver.adaptivity = True

solver.max_iterations = 50000
solver.tolerance = 1.0e-12

# Run the problem
solver.initialize()
solver.execute(verbose=1)

out = solver.outputs
sim_times = np.round(out.times, 6)
grid = [p[2] for p in out.grid]

plt.figure()
times = [0.0, 0.6, 1.1, 1.7]
for t in times:
    i = np.argmin(abs(sim_times - t))
    phi = out.flux[i][0]
    plt.plot(grid, phi, label=f"Time = {t:.2g} sec")
plt.legend()
plt.grid(True)

plt.figure()
plt.title("Power Profile")
plt.plot(sim_times, out.power)
plt.grid(True)

plt.show()
