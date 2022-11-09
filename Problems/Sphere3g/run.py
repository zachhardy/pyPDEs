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
    description="The three group sphere problem.",
    formatter_class=CustomFormatter
)

parser.add_argument("--n_cells", default=100, type=int,
                    help="The number of cells.")

parser.add_argument("--t_end", default=0.1, type=float,
                    help="The simulation end time.")

parser.add_argument("--dt", default=0.002, type=float,
                    help="The time step size.")

parser.add_argument("--radius", default=6.0, type=float,
                    help="The sphere radius.")

parser.add_argument("--density", default=0.05, type=float,
                    help="The number density in atoms/b-cm.")

parser.add_argument("--down_scatter", default=1.46, type=float,
                    help="The down-scattering cross-section in b.")

parser.add_argument("--output_directory",
                    default=f"{path}/reference", type=str,
                    help="The output directory.")

argv = parser.parse_args()


# ==================================================
# Initial condition function
# ==================================================

def ic(r: CartesianVector) -> float:
    return 0.0  # - r.z ** 2 / argv.radius ** 2


# ==================================================
# Create the spatial mesh
# ==================================================

mesh = create_1d_orthomesh(
    [0.0, argv.radius], [argv.n_cells], [0], "SPHERICAL"
)
fv = FiniteVolume(mesh)

# ==================================================
# Create the materials
# ==================================================

material = Material()

xs = CrossSections()
xs.read_xs_file(f"{path}/xs/Sphere3g.xs", argv.density)
xs.transfer_matrices[0][1][0] = argv.down_scatter * argv.density
material.properties.append(xs)

# ==================================================
# Boundary conditions
# ==================================================

def bndry_src(r: CartesianVector, t: float) -> float:
    return 1.0e8 if t <= argv.dt else 0.0


boundary_vals = [{0: bndry_src}]
boundary_info = [("REFLECTIVE", -1), ("MARSHAK", 0)]

# ==================================================
# Create the solver
# ==================================================

solver = TransientSolver(fv, [material], boundary_info, boundary_vals)

# ==================================================
# Temporal discretization
# ==================================================

solver.initial_conditions = {0: ic}  # , 1: ic}

solver.normalization_method = None
solver.scale_fission_xs = False

solver.t_end = argv.t_end
solver.dt = argv.dt
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
solver.output_directory = argv.output_directory

# ==================================================
# Execute
# ==================================================

solver.initialize()
solver.execute()
