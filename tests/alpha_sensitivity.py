import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *


def analytic_alpha(n: List[int]) -> float:
    return ex.get_mode(n, method='eig').alpha.real


abs_path = os.path.dirname(os.path.abspath(__file__))

path = os.path.dirname(os.path.abspath(__file__))
path += '/three_group_sphere/transient/sphere6cm.obj'
ex: AnalyticSolution = load(path)

# Mesh parameters
r_b = 6.0
mesh = create_1d_mesh([0.0, 6.0], [100],
                      coord_sys='spherical')
discretization = FiniteVolume(mesh)

# Create cross sections
material = Material()
xs = CrossSections()

# Read in the correct cross sections
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)

material.add_properties(xs)

# Create boundary conditions
boundaries = [ZeroFluxBoundary(xs.n_groups),
              ZeroFluxBoundary(xs.n_groups)]

# Initialize forward problem and attach objects
solver = AlphaEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = [material]
solver.boundaries = boundaries
solver.initial_condition = \
    [lambda r: 1.0 - r ** 2 / r_b ** 2,
     lambda r: 0.0,
     lambda r: 0.0]

print(solver.compute_k_sensitivity())
print(solver.compute_alpha_sensitivity()[:10].real)

plt.show()
