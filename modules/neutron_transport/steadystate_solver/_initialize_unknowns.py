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

    # Initialize flux moments
    n_nodes = self.discretization.n_dofs()
    n_phi_dofs = n_nodes * self.n_moments * self.n_groups

    self.phi = np.zeros(n_phi_dofs)
    self.phi_prev = np.zeros(n_phi_dofs)
    self.q_moments = np.zeros(n_phi_dofs)

    # Initialize anglular fluxes
    n_psi_dofs = n_nodes * self.n_angles * self.n_groups
    self.psi = np.zeros(n_psi_dofs)

    n_cells = self.mesh.n_cells
    n_faces = len(self.mesh.cells[0].faces)
    shape = (self.n_angles, self.n_groups, n_cells, n_faces, 1)
    self.psi_interface = np.zeros(shape=shape)

    # Initialize precursors
    if self.use_precursors:
        n_precursor_dofs = self.mesh.n_cells * self.max_precursors
        self.precursors = np.zeros(n_precursor_dofs)
