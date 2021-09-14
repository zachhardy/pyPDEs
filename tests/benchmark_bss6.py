import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from typing import List

from pyPDEs.mesh import create_1d_mesh
from pyPDEs.spatial_discretization import *
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.utilities.boundaries import *

from modules.neutron_diffusion import *

# Create mesh and discretization
zones = [0.0, 40.0, 200.0, 240.0]
n_cells = [40, 160, 40]
material_ids = [0, 1, 2]
mesh = create_1d_mesh(zones, n_cells, material_ids, coord_sys="CARTESIAN")
discretization = FiniteVolume(mesh)
