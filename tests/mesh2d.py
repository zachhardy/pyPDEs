import numpy as np

from pyPDEs.mesh import Mesh, Cell, Face
from pyPDEs.mesh import create_2d_mesh
from pyPDEs.utilities.vector import Vector

x = np.linspace(0.0, 1.0, 3)
y = np.linspace(0.0, 1.0, 3)

mesh = create_2d_mesh(x, y, verbose=True)
