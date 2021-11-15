import os
import sys
import itertools
import numpy as np

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import *
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f"rm -r {path}/*")


# Define current directory
script_path = os.path.dirname(os.path.abspath(__file__))


# Get inputs
with_ics = True
with_size = False
for arg in sys.argv:
    if "with_ics" in arg:
        with_ics = bool(int(arg.split("=")[1]))
    if "with_size" in arg:
        with_size = bool(int(arg.split("=")[1]))

study_name = "density"
study_name += "_size" if with_size else ""
study_name += "_ics" if with_ics else "_k"

# Define parameter space
parameters = {}
if with_size:
    densities = np.linspace(0.0495, 0.0505, 6)
    parameters["density"] = list(np.round(densities, 6))

    sizes = np.linspace(5.95, 6.05, 6)
    parameters["size"] = list(np.round(sizes, 6))
else:
    densities = np.linspace(0.047, 0.053, 31)
    parameters["density"] = list(np.round(densities, 6))

keys = list(parameters.keys())
values = list(itertools.product(*parameters.values()))


# Setup the problem
mesh = create_1d_mesh([0.0, 6.0], [100], coord_sys="SPHERICAL")
discretization = FiniteVolume(mesh)

material = Material()
xs = CrossSections()
xs.read_from_xs_file("xs/three_grp_us.cxs", density=0.05)
material.add_properties(xs)

boundaries = [ReflectiveBoundary(np.zeros(xs.n_groups)),
              ZeroFluxBoundary(np.zeros(xs.n_groups))]

solver = TransientSolver()
solver.mesh = mesh
solver.discretization = discretization
solver.boundaries = boundaries
solver.materials = [material]
solver.use_precursors = False

rf = mesh.vertices[-1].z
ics = [lambda r: 1.0 - r**2 / rf**2,
       lambda r: 1.0 - r**2 / rf**2,
       lambda r: 0.0]
solver.initial_conditions = ics if with_ics else None
solver.normalize_fission = False

solver.t_final = 0.1
solver.dt = 2.0e-3
solver.stepping_method = "tbdf"

solver.write_outputs = True

# Define output path
output_path = os.path.join(script_path, "outputs", study_name)
setup_directory(output_path)

# Save parameters
param_filepath = os.path.join(output_path, "params.txt")
np.savetxt(param_filepath, np.array(values), fmt="%.8e")

# Run the reference problem
msg = "===== Running reference ====="
head = "=" * len(msg)
print()
print("\n".join([head, msg, head]))

simulation_path = os.path.join(output_path, "reference")
setup_directory(simulation_path)
solver.output_directory = simulation_path
solver.initialize()
solver.execute()

# Run the study
for n, params in enumerate(values):

    # Setup output path
    simulation_path = os.path.join(output_path, str(n).zfill(3))
    setup_directory(simulation_path)
    solver.output_directory = simulation_path

    # Modify system parameters
    if "density" in keys:
        ind = keys.index("density")
        xs.read_from_xs_file("xs/three_grp_us.cxs", density=params[ind])
    if "size" in keys:
        ind = keys.index("size")
        solver.mesh = create_1d_mesh([0.0, params[ind]], [mesh.n_cells],
                                     coord_sys=mesh.coord_sys)
        solver.discretization = FiniteVolume(mesh)

    # Run the problem
    solver.initialize()

    msg = f"===== Running simulation {n} ====="
    head = "=" * len(msg)
    print("\n".join(["", head, msg, head]))
    for p in range(len(params)):
        pname = keys[p].capitalize()
        print(f"{pname:<10}:\t{params[p]:<5}")
    print(f"{'k_eff':<10}:\t{solver.k_eff:<8.5f}")

    solver.execute()
