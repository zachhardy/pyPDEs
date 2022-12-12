import os
import sys
import argparse
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pyPDEs.mesh import create_1d_orthomesh
from pyPDEs.mesh import CartesianVector
from pyPDEs.math.discretization import FiniteVolume

from pyPDEs.material import Material
from pyPDEs.material import CrossSections
from pyPDEs.material import IsotropicMultiGroupSource

from modules.neutron_diffusion import AlphaEigenvalueSolver
from modules.neutron_diffusion import KEigenvalueSolver

from readers.neutronics import NeutronicsSimulationReader

path = os.path.abspath(os.path.dirname(__file__))


# ==================================================
# Parse Arguments
# ==================================================

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.MetavarTypeHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description=textwrap.dedent('''\
    The alpha-eigenvalue problem for the three-group sphere problem.
    '''),
    formatter_class=CustomFormatter
)

parser.add_argument("--n_cells", default=100, type=int,
                    help="The number of cells.")

parser.add_argument("--radius", default=4.2, type=float,
                    help="The sphere radius.")

parser.add_argument("--density", default=0.05, type=float,
                    help="The number density in atoms/b-cm.")

parser.add_argument("--down_scatter", default=1.46, type=float,
                    help="The down-scattering cross-section in b.")

parser.add_argument("--n_modes", default=-1, type=int,
                    help="The number of alpha-eigenfunctions to keep.")

parser.add_argument("--output_directory",
                    default=f"{path}/reference", type=str,
                    help="The output directory.")

argv = parser.parse_args()

# ==================================================
# Initial condition function
# ==================================================

reader = NeutronicsSimulationReader(f"{path}/reference").read()

def ic_func(r: CartesianVector) -> float:
    return 0.0  # - r.z ** 2 / argv.radius ** 2


ic = {0: ic_func}
ic = reader.get_flux_moment_snapshot(reader.times[10])

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

boundary_info = [("REFLECTIVE", -1), ("VACUUM", -1)]

# ==================================================
# Create the solver
# ==================================================

alpha_solver = AlphaEigenvalueSolver(
    fv, [material], boundary_info, n_modes=argv.n_modes, fit_data=ic
)

k_solver = KEigenvalueSolver(fv, [material], boundary_info)

# ==================================================
# Set options
# ==================================================

alpha_solver.write_outputs = True
alpha_solver.output_directory = argv.output_directory

# ==================================================
# Execute
# ==================================================

k_solver.initialize()
k_solver.execute()

alpha_solver.initialize()
alpha_solver.execute()

plt.show()
