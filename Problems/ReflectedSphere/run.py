import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_orthomesh
from pyPDEs.mesh import CartesianVector
from pyPDEs.math.discretization import FiniteVolume

from pyPDEs.material import Material
from pyPDEs.material import CrossSections
from pyPDEs.material import IsotropicMultiGroupSource

from modules.neutron_diffusion import SteadyStateSolver
from modules.neutron_diffusion import KEigenvalueSolver
from modules.neutron_diffusion import TransientSolver

path = os.path.abspath(os.path.dirname(__file__))

# ==================================================
# Parse Arguments
# ==================================================

radius = 5.0
thickness = 1.0
density = 0.05
sig_s01 = 1.46

xsdir = os.path.join(path, "xs")
outdir = os.path.join(path, "reference")

for i, arg in enumerate(sys.argv[1:]):
    print(f"Parsing argument {i}: {arg}")

    value = arg.split("=")[1]
    if arg.find("radius") == 0:
        radius = float(value)
    elif arg.find("thickness") == 0:
        thickness = float(thickness)
    elif arg.find("density") == 0:
        density = float(value)
    elif arg.find("scatter") == 0:
        sig_s01 = float(value)
    elif arg.find("output_directory") == 0:
        outdir = os.path.join(path, value)
    elif arg.find("xs_directory") == 0:
        xsdir = value

# ==================================================
# Initial condition function
# ==================================================


def ic(r):
    assert isinstance(r, CartesianVector)
    r_b = mesh.vertices[-1]
    return 1.0 - r.z ** 2 / r_b.z ** 2
    # return 1.0

# ==================================================
# Create the spatial mesh
# ==================================================

mesh = create_1d_orthomesh([0.0, radius, radius + thickness],
                           [100, 20], [0, 1], "SPHERICAL")
# mesh = create_1d_orthomesh([0.0, radius], [100], [0], "SPHERICAL")
fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

materials = [Material(), Material()]
xsecs = [CrossSections(), CrossSections()]
xs_paths = [f"{xsdir}/fuel.xs", f"{xsdir}/reflector.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    rho = density if i == 0 else 0.124
    xs.read_xs_file(xs_path, rho)
    for g in range(xs.n_groups):
        xs.inv_velocity[g] *= 1.0e6
    material.properties.append(xs)

# material = Material()
# xs = CrossSections()
# xs.read_xs_file(f"{xsdir}/fuel.xs", density)
# xs.inv_velocity *= 1.0e6
# material.properties.append(xs)
# materials = [material]

# ==================================================
# Boundary conditions
# ==================================================

boundary_info = [("REFLECTIVE", -1), ("ZERO_FLUX", -1)]

# ==================================================
# Create the solver
# ==================================================

solver = TransientSolver(fv, materials, boundary_info)

# ==================================================
# Temporal discretization
# ==================================================

solver.initial_conditions = {0: ic, 1: ic}

solver.normalization_method = "TOTAL_POWER"
solver.scale_fission_xs = False

solver.t_end = 10.0
solver.dt = 0.1
solver.time_stepping_method = "TBDF2"

# ==================================================
# Set options
# ==================================================

solver.use_precursors = False
solver.lag_precursors = False

solver.adaptive_time_stepping = True
solver.refine_threshold = 0.05
solver.coarsen_threshold = 0.01

solver.write_outputs = True
solver.output_directory = outdir

# ==================================================
# Execute
# ==================================================

solver.initialize()
solver.execute()
