import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_2d_orthomesh
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

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description="The TWIGL benchmark problem.",
    formatter_class=CustomFormatter
)

parser.add_argument("--t_end", default=0.5, type=float,
                    help="The simulation end time.")

parser.add_argument("--dt", default=0.01, type=float,
                    help="The time step size.")

parser.add_argument("--magnitude", default=-0.02333, type=float,
                    help="The cross-section ramp magnitude.")

parser.add_argument("--duration", default=0.2, type=float,
                    help="The cross-section ramp duration.")

parser.add_argument("--down_scatter", default=0.01, type=float,
                    help="The down-scattering cross-section in 1/cm.")

parser.add_argument("--output_directory",
                    default=f"{path}/reference", type=str,
                    help="The output directory.")

argv = parser.parse_args()

# ==================================================
# Cross-section function
# ==================================================


def f(group_num, args, reference):
    t = args[0]
    if group_num == 1:
        if 0.0 < t <= argv.duration:
            return (1.0 + t / argv.duration * argv.magnitude) * reference
        else:
            return (1.0 + argv.magnitude) * reference
    else:
        return reference


# ==================================================
# Create the spatial mesh
# ==================================================

# Load a pickled mesh
if os.path.isfile(path):
    print(f"Using a mesh pickled at {path}/mesh.obj.")
    with open(f"{path}/mesh.obj", 'rb') as file:
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

    with open(f"{path}/mesh.obj", 'wb') as file:
        pickle.dump(mesh, file)

fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

materials = [Material() for _ in range(3)]
xsecs = [CrossSections() for _ in range(3)]
xs_paths = [f"{path}/xs/fuel0.xs",
            f"{path}/xs/fuel0.xs",
            f"{path}/xs/fuel1.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    xs.read_xs_file(xs_path)
    xs.sigma_a_function = f if i == 0 else None
    if i == 2:
        xs.transfer_matrices[0][1][0] = argv.down_scatter
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

solver.t_end = argv.t_end
solver.dt = argv.dt
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
solver.output_directory = argv.output_directory

# ==================================================
# Execute
# ==================================================

solver.initialize()
solver.execute()
