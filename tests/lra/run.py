import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from time import time
from copy import deepcopy

from pyPDEs.mesh import create_2d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

from xs import *

abs_path = os.path.dirname(os.path.abspath(__file__))

# Create mesh, assign material IDs
x_verts = np.linspace(0.0, 165.0, 23)
y_verts = np.linspace(0.0, 165.0, 23)
mesh = create_2d_mesh(x_verts, y_verts, verbose=True)

for cell in mesh.cells:
    c = cell.centroid
    if (0.0 <= c.x <= 15.0 or 75.0 <= c.x <= 105.0) and \
            (0.0 <= c.y <= 15.0 or 75.0 <= c.y <= 105.0):
        cell.material_id = 1
    elif 0.0 <= c.x <= 105.0 and 0.0 <= c.y <= 105.0 and \
            cell.material_id == -1:
        cell.material_id = 0
    elif 0.0 <= c.x <= 105.0 and 105.0 <= c.y <= 135.0:
        cell.material_id = 2
    elif 105.0 <= c.x <= 135.0 and 0.0 <= c.y <= 75.0:
        cell.material_id = 2
    elif 105.0 <= c.x <= 135.0 and 75.0 <= c.y <= 105.0:
        cell.material_id = 3
    elif 105.0 <= c.x <= 120.0 and 105.0 <= c.y <= 120.0:
        cell.material_id = 4
    else:
        cell.material_id = 5

# Create discretizations
discretization = FiniteVolume(mesh)

# Create materials
materials = []
materials.append(Material("Fuel 1 with Rod"))
materials.append(Material("Fuel 1 without Rod"))
materials.append(Material("Fuel 2 with Rod"))
materials.append(Material("Rod Ejection Region"))
materials.append(Material("Fuel 2 without Rod"))
materials.append(Material("Reflector"))

xs = [CrossSections() for _ in range(len(materials))]
data = [fuel_1_with_rod, fuel_1_without_rod,
        fuel_2_with_rod, fuel_2_with_rod, fuel_2_without_rod,
        reflector]
fcts = [sigma_a_without_rod, sigma_a_without_rod, sigma_a_without_rod,
        sigma_a_with_rod, sigma_a_without_rod, None]
for i in range(len(materials)):
    xs[i].read_from_xs_dict(data[i])
    xs[i].sigma_a_function = fcts[i]
    materials[i].add_properties(xs[i])

# Create boundary conditions
n_groups = xs[0].n_groups
boundaries = [ReflectiveBoundary(n_groups),
              ZeroFluxBoundary(n_groups),
              ZeroFluxBoundary(n_groups),
              ReflectiveBoundary(n_groups)]

# Initialize solver and attach objects
solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = materials
solver.boundaries = boundaries

# Iterative parameters
solver.tolerance = 1.0e-12
solver.max_iterations = int(1.0e4)

solver.is_nonlinear = True
solver.nonlinear_tolerance = 1.0e-8
solver.nonlinear_max_iterations = 20

# Set precursor options
solver.use_precursors = True
solver.lag_precursors = False

# Set feedback options
solver.feedback_coeff = 3.034e-3
solver.energy_per_fission = 3.204e-11
solver.conversion_factor= 3.83e-11

# Initial power
solver.initial_power = 1.0e-6
solver.phi_norm_method = "AVERAGE"

# Set time stepping options
solver.t_final = 3.0
solver.dt = 0.01
solver.method = "TBDF2"

solver.adaptivity = True
solver.refine_level = 0.1
solver.coarsen_level = 0.01

# Output informations
solver.write_outputs = True
solver.output_directory = \
    os.path.join(abs_path, "outputs")

# Run the problem
solver.initialize(verbose=1)
solver.execute(verbose=1)
