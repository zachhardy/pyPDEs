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

r_inner = 4.25
r_outer = 5.25
density = 0.05

xsdir = os.path.join(path, "xs")
outdir = os.path.join(path, "reference")

for i, arg in enumerate(sys.argv[1:]):
    print(f"Parsing argument {i}: {arg}")

    value = arg.split("=")[1]
    if arg.find("r_inner") == 0:
        r_inner = float(value)
    elif arg.find("r_outer") == 0:
        r_outer = float(value)
    elif arg.find("density") == 0:
        density = float(value)
    elif arg.find("output_directory") == 0:
        outdir = os.path.join(path, value)
    elif arg.find("xs_directory") == 0:
        xsdir = value

# ==================================================
# Create the spatial mesh
# ==================================================

mesh = create_1d_orthomesh([0.0, r_inner, r_outer],
                           [100, 20], [0, 1], "SPHERICAL")
# mesh = create_1d_orthomesh([0.0, radius], [100], [0], "SPHERICAL")
fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

materials = [Material(), Material()]
xsecs = [CrossSections(), CrossSections()]
xs_paths = [f"{xsdir}/fuel.xs", f"{xsdir}/reflector.xs"]

# materials = [Material()]
# xsecs = [CrossSections()]
# xs_paths = [f"{xsdir}/fuel.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    rho = density if i == 0 else 0.124
    xs.read_xs_file(xs_path, rho)
    for g in range(xs.n_groups):
        xs.inv_velocity[g] *= 1.0e6
    material.properties.append(xs)

# ==================================================
# Temporal discretization
# ==================================================

def ic(r: CartesianVector) -> float:
    return 0.0

initial_conditions = {}

normalization_method = None #"TOTAL_POWER"
scale_fission_xs = False

t_end = 1.0
dt = 0.01
time_stepping_method = "TBDF2"

# ==================================================
# Boundary conditions
# ==================================================

def boundary_source(r: CartesianVector, t: float = 0.0) -> float:
    r_b = mesh.vertices[-1]
    if r.z == r_b.z and 0 < t <= dt:
        return 1.0e10
    else:
        return 0.0

boundary_vals = [[[boundary_source], [0.0], [0.0]]]
boundary_info = [("REFLECTIVE", -1), ("MARSHAK", 0)]

# ==================================================
# Create the solver
# ==================================================

solver = TransientSolver(fv, materials, boundary_info, boundary_vals)

solver.initial_conditions = initial_conditions

solver.normalization_method = normalization_method
solver.scale_fission_xs = scale_fission_xs

solver.t_end = t_end
solver.dt = dt
solver.time_stepping_method = time_stepping_method

solver.use_precursors = False
solver.lag_precursors = False

solver.adaptive_time_stepping = False
solver.refine_threshold = 0.05
solver.coarsen_threshold = 0.01

solver.write_outputs = True
solver.output_directory = outdir

# ==================================================
# Execute
# ==================================================

solver.initialize()
solver.execute()
