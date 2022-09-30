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

magnitude = 0.97667 - 1.0
duration = 0.2
sig_s01 = 0.01

xsdir = os.path.join(path, "xs")
outdir = os.path.join(path, "outputs")

for i, arg in enumerate(sys.argv[1:]):
    print(f"Parsing argument {i}: {arg}")

    value = arg.split("=")[1]
    if arg.find("magnitude") == 0:
        magnitude = float(value)
    elif arg.find("duration") == 0:
        duration = float(value)
    elif arg.find("scatter") == 0:
        sig_s01 = float(value)
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
            return (1.0 + t/duration * magnitude) * reference
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
    verts = np.linspace(0.0, 80.0, 41)
    mesh = create_2d_orthomesh(verts, verts)

    for cell in mesh.cells:
        c = cell.centroid
        if 24.0 <= c.x <= 56.0 and 24.0 <= c.y <= 56.0:
            cell.material_id = 0
        elif 0.0 <= c.x <= 24.0 and 24.0 <= c.y <= 56.0:
            cell.material_id = 1
        elif 24.0 <= c.x <= 56.0 and 0 <= c.y <= 24.0:
            cell.material_id = 1
        else:
            cell.material_id = 2

    with open(path, 'wb') as file:
        pickle.dump(mesh, file)

fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

materials = [Material() for _ in range(3)]
xsecs = [CrossSections() for _ in range(3)]
xs_paths = [f"{xsdir}/fuel0.xs",
            f"{xsdir}/fuel0.xs",
            f"{xsdir}/fuel1.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    xs.read_xs_file(xs_path)
    xs.sigma_a_function = f if i == 0 else None
    if i == 2:
        xs.transfer_matrices[0][1][0] = sig_s01
    material.properties.append(xs)

# ==================================================
# Boundary conditions
# ==================================================

boundary_info = [("REFLECTIVE", -1), ("VACUUM", -1),
                 ("VACUUM", -1), ("REFLECTIVE", -1)]

# ==================================================
# Create the solver
# ==================================================

solver = TransientSolver(fv, materials, boundary_info)

# ==================================================
# Temporal discretization
# ==================================================

solver.normalization_method = "TOTAL_POWER"
solver.scale_fission_xs = True

solver.t_end = 0.5
solver.dt = 1.0e-2
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
