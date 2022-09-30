import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pyPDE.mesh import create_1d_orthomesh
from pyPDE.mesh import CartesianVector
from pyPDE.math.discretization import FiniteVolume

from pyPDE.material import Material
from pyPDE.material import CrossSections
from pyPDE.material import IsotropicMultiGroupSource

from modules.neutron_diffusion import SteadyStateSolver
from modules.neutron_diffusion import KEigenvalueSolver
from modules.neutron_diffusion import TransientSolver


path = os.path.abspath(os.path.dirname(__file__))


# ==================================================
# Parse Arguments
# ==================================================

magnitude = -0.01
duration = 1.0
interface = 40.0

xsdir = os.path.join(path, "xs")
outdir = os.path.join(path, "outputs")

for i, arg in enumerate(sys.argv[1:]):
    print(f"Parsing argument {i}: {arg}")

    value = arg.split("=")[1]
    if arg.find("magnitude") == 0:
        magnitude = float(value)
    elif arg.find("duration") == 0:
        duration = float(value)
    elif arg.find("interface") == 0:
        interface = float(value)
    elif arg.find("output_directory") == 0:
        outdir = value
    elif arg.find("xs_directory") == 0:
        xsdir = value


# ==================================================
# Cross-section function
# ==================================================


def f(group_num, args, reference):
    t = args[0]
    if group_num == 1:
        if 0.0 < t <= duration:
            return (1.0 + t / duration * magnitude) * reference
        else:
            return (1.0 + magnitude) * reference
    else:
        return reference


# ==================================================
# Create the spatial mesh
# ==================================================

zone_edges = [0.0, interface, 200.0, 240.0]
zone_subdivs = [20, 80, 20]
material_ids = [0, 1, 2]
mesh = create_1d_orthomesh(zone_edges, zone_subdivs, material_ids)
fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

materials = [Material() for _ in range(3)]
xsecs = [CrossSections() for _ in range(3)]
xs_paths = [f"{path}/xs/fuel0.xs",
            f"{path}/xs/fuel1.xs",
            f"{path}/xs/fuel0.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    xs.read_xs_file(xs_path)
    xs.sigma_a_function = f if i == 0 else None
    material.properties.append(xs)

# ==================================================
# Boundary conditions
# ==================================================

boundary_info = [("ZERO_FLUX", -1), ("ZERO_FLUX", -1)]

# ==================================================
# Create the solver
# ==================================================

solver = TransientSolver(fv, materials, boundary_info)

# ==================================================
# Temporal discretization
# ==================================================

solver.normalization_method = "TOTAL_POWER"
solver.scale_fission_xs = True

solver.t_end = 2.0
solver.dt = 0.04
solver.time_stepping_method = "TBDF2"

# ==================================================
# Set options
# ==================================================

solver.use_precursors = True
solver.lag_precursors = False

solver.tolerance = 1.0e-10
solver.max_iterations = 1000

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
