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

# ------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description="The LRA benchmark problem.",
    formatter_class=CustomFormatter
)

parser.add_argument("--t_end", default=3.0, type=float,
                    help="The simulation end time.")

parser.add_argument("--dt", default=0.01, type=float,
                    help="The time step size.")

parser.add_argument("--magnitude", default=-0.1212369, type=float,
                    help="The cross-section ramp magnitude.")

parser.add_argument("--duration", default=2.0, type=float,
                    help="The cross-section ramp magnitude.")

parser.add_argument("--gamma", default=3.034e-3, type=float,
                    help="The temperature feedback coefficient.")

parser.add_argument("--output_directory",
                    default=f"{path}/reference", type=str,
                    help="The output directory.")

argv = parser.parse_args()

# ------------------------------------------------------------
# Cross-Section Functions
# ------------------------------------------------------------

T0 = 300.0


def feedback_func(group_num, args, reference):
    t, T = args
    if group_num == 0:
        return (1.0 + argv.gamma * (np.sqrt(T) - np.sqrt(T0))) * reference
    else:
        return reference


def f(group_num, args, reference):
    t, T = args
    if group_num == 0:
        return feedback_func(group_num, args, reference)
    elif group_num == 1:
        if 0.0 < t <= argv.duration:
            return (1.0 + t / argv.duration * argv.magnitude) * reference
        else:
            return (1.0 + argv.magnitude) * reference
    else:
        return reference


# ------------------------------------------------------------
# Mesh
# ------------------------------------------------------------

# Load a pickled mesh
if os.path.isfile(path):
    print(f"Using a mesh pickled at {path}/mesh.obj.")
    with open(f"{path}/mesh.obj", 'rb') as file:
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

    with open(f"{path}/mesh.obj", 'wb') as file:
        pickle.dump(mesh, file)

fv = FiniteVolume(mesh)

# ------------------------------------------------------------
# Materials
# ------------------------------------------------------------

materials = [Material() for _ in range(6)]
xsecs = [CrossSections() for _ in range(6)]
xs_paths = [f"{path}/xs/fuel0_w_rod.xs",
            f"{path}/xs/fuel0_wo_rod.xs",
            f"{path}/xs/fuel1_w_rod.xs",
            f"{path}/xs/fuel1_wo_rod.xs",
            f"{path}/xs/fuel1_w_rod.xs",
            f"{path}/xs/reflector.xs"]

it = zip(materials, xsecs, xs_paths)
for i, (material, xs, xs_path) in enumerate(it):
    xs.read_xs_file(xs_path)
    xs.sigma_a_function = f if i == 4 else feedback_func
    material.properties.append(xs)

# ------------------------------------------------------------
# Boundary Conditions
# ------------------------------------------------------------

boundary_info = [("REFLECTIVE", -1), ("ZERO_FLUX", -1),
                 ("ZERO_FLUX", -1), ("REFLECTIVE", -1)]

# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------

solver = TransientSolver(fv, materials, boundary_info)

# ------------------------------------------------------------
# Temporal Discretization
# ------------------------------------------------------------

solver.initial_power = 1.0e-6
solver.normalization_method = "AVERAGE_POWER_DENSITY"
solver.scale_fission_xs = True

solver.t_end = argv.t_end
solver.dt = argv.dt
solver.time_stepping_method = "TBDF2"

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------

solver.use_precursors = True
solver.lag_precursors = False

solver.tolerance = 1.0e-8
solver.max_iterations = 500

solver.adaptive_time_stepping = False
solver.refine_threshold = 0.1
solver.coarsen_threshold = 0.01

solver.write_outputs = True
solver.output_directory = argv.output_directory

# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------

solver.initialize()
solver.execute()
