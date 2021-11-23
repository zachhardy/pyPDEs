import os
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
x_verts = np.linspace(0.0, 10.0, 11)
y_verts = np.linspace(0.0, 10.0, 11)

mesh = create_2d_mesh(x_verts, y_verts, verbose=True)
for cell in mesh.cells:
    cell.material_id = 0

# Create discretizations
discretization = FiniteVolume(mesh)

# Create materials
material = Material()

xs = CrossSections()
xs.read_from_xs_file('xs/fuel_1g.cxs')
material.add_properties(xs)

# Create boundary conditions
n_groups = xs.n_groups
boundaries = [ReflectiveBoundary(n_groups),
              VacuumBoundary(n_groups),
              ReflectiveBoundary(n_groups),
              VacuumBoundary(n_groups)]

# Initialize solver and attach objects
solver = KEigenvalueSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.materials = [material]
solver.boundaries = boundaries

solver.tolerance = 1.0e-8
solver.max_iterations = 1000

# Set options
solver.use_precursors = True
solver.lag_precursors = False

# Set time stepping options
solver.t_final = 0.5
solver.dt = 1.0e-2
solver.method = 'tbdf2'

# Output informations
solver.write_outputs = True
solver.output_directory = \
    os.path.join(abs_path, 'outputs/ramp')

# Run the problem
solver.initialize(verbose=1)
solver.execute(verbose=1)
solver.plot_flux()

plt.figure()
plt.spy(solver.assemble_matrix())
plt.show()
