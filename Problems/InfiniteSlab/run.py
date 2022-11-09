import os
import sys
import argparse
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

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description="The supercritical infinite slab problem.",
    formatter_class=CustomFormatter
)

parser.add_argument("--t_end", default=2.0, type=float,
                    help="The simulation end time.")

parser.add_argument("--dt", default=0.04, type=float,
                    help="The time step size.")

parser.add_argument("--magnitude", default=-0.01, type=float,
                    help="The cross-section ramp magnitude.")

parser.add_argument("--duration", default=1.0, type=float,
                    help="The duration of the cross-section ramp.")

parser.add_argument("--interface", default=40.0, type=float,
                    help="The interface location in cm.")

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

zone_edges = [0.0, argv.interface, 200.0, 240.0]
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
