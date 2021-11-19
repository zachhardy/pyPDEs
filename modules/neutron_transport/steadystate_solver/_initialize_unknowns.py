import numpy as np
from numpy import ndarray

from pyPDEs.material import *
from pyPDEs.utilities import UnknownManager
from ..boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def initialize_unknowns(self: 'SteadyStateSolver') -> None:
    """
    Initialize the unknown storage for the problem.
    """
    # Initialize unknown managers
    self.phi_uk_man = UnknownManager()
    for m in range(self.n_moments):
        self.phi_uk_man.add_unknown(self.n_groups)

    self.psi_uk_man = UnknownManager()
    for n in range(self.n_angles):
        self.psi_uk_man.add_unknown(self.n_groups)

    # Initialize vectors
    n_nodes = self.discretization.n_dofs()
    n_phi_dofs = n_nodes * self.n_groups * self.n_moments
    n_psi_dofs = n_nodes * self.n_groups * self.n_angles

    self.phi = np.zeros(n_phi_dofs)
    self.phi_prev = np.zeros(n_phi_dofs)
    self.psi = np.zeros(n_psi_dofs)
    if self.use_precursors:
        n_precursors_dofs = self.n_precursors * self.mesh.n_cells
        self.precursors = np.zeros(n_precursors_dofs)
