import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pyPDE.mesh import create_2d_orthomesh
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

magnitude = 0.8787631 - 1.0
duration = 2.0
gamma = 3.034e-3

xsdir = os.path.join(path, "xs")
outdir = os.path.join(path, "outputs")

for i, arg in enumerate(sys.argv[1:]):
    print(f"Parsing argument {i}: {arg}")

    value = arg.split("=")[1]
    if arg.find("magnitude") == 0:
        magnitude = float(value)
    elif arg.find("duration") == 0:
        duration = float(value)
    elif arg.find("feedback") == 0:
        gamma = float(value)
    elif arg.find("output_directory") == 0:
        outdir = value
    elif arg.find("xs_directory") == 0:
        xsdir = value

# ==================================================
# Cross-section function
# ==================================================

T0 = 300.0


def feedback_func(group_num, args, reference):
    t, T = args
    if group_num == 0:
        return (1.0 + gamma * (np.sqrt(T) - np.sqrt(T0))) * reference
    else:
        return reference


def f(group_num, args, reference):
    t, T = args
    if group_num == 0:
        return feedback_func(group_num, args, reference)
    elif group_num == 1:
        if 0.0 < t <= duration:
            return (1.0 + t / duration * magnitude) * reference
        else:
            return (1.0 + magnitude) * reference
    else:
        return reference


# ==================================================
# Create the spatial mesh
# ==================================================

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, "mesh.obj")

# Load a pickled mesh
if os.path.isfile(path):
    print(f"Using a mesh pickled at {path}.")
    with open(path, 'rb') as file:
        mesh = pickle.load(file)

# Define the mesh and pickle it
else:
    verts = np.linspace(0.0, 165.0, 23)
    mesh = create_2d_orthomesh(verts, verts)

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
        elif 105.0 <= c.x <= 120.0 and 105.0 <= c.y <= 120.0:
            cell.material_id = 3
        elif 105.0 <= c.x <= 135.0 and 75.0 <= c.y <= 105.0:
            cell.material_id = 4
        else:
            cell.material_id = 5

    with open(path, 'wb') as file:
        pickle.dump(mesh, file)

fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

materials = [Material() for _ in range(6)]
xsecs = [CrossSections() for _ in range(6)]
xs_paths = [f"{xsdir}/fuel0_w_rod.xs",
            f"{xsdir}/fuel0_wo_rod.xs",
            f"{xsdir}/fuel1_w_rod.xs",
            f"{xsdir}/fuel1_wo_rod.xs",
            f"{xsdir}/fuel1_w_rod.xs",
            f"{xsdir}/reflector.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    xs.read_xs_file(xs_path)
    xs.sigma_a_function = f if i == 4 else feedback_func
    material.properties.append(xs)

# ==================================================
# Boundary conditions
# ==================================================

boundary_info = [("REFLECTIVE", -1), ("ZERO_FLUX", -1),
                 ("ZERO_FLUX", -1), ("REFLECTIVE", -1)]

# ==================================================
# Create the solver
# ==================================================

solver = TransientSolver(fv, materials, boundary_info)

# ==================================================
# Temporal discretization
# ==================================================

solver.initial_power = 1.0e-6
solver.normalization_method = "AVERAGE_POWER_DENSITY"
solver.scale_fission_xs = True

solver.t_end = 3.0
solver.dt = 1.0e-2
solver.time_stepping_method = "TBDF2"

# ==================================================
# Set options
# ==================================================

solver.use_precursors = True
solver.lag_precursors = False

solver.tolerance = 1.0e-8
solver.max_iterations = 500

solver.adaptive_time_stepping = False
solver.refine_threshold = 0.1
solver.coarsen_threshold = 0.01

solver.write_outputs = True
solver.output_directory = outdir

# ==================================================
# Execute
# ==================================================

solver.initialize()
solver.execute()
