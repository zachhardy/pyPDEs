import numpy as np
from numpy import ndarray

from pyPDEs.material import *
from pyPDEs.utilities import UnknownManager

from ..boundaries import *
from ..data_structures import *

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
    phi_shape = (self.n_moments, self.n_groups, n_nodes)

    self.phi = np.zeros(phi_shape)
    self.phi_prev = np.zeros(phi_shape)
    self.q_moments = np.zeros(phi_shape)

    # Initialize anglular fluxes
    psi_shape = (self.n_angles, self.n_groups, n_nodes)
    self.psi = np.zeros(psi_shape)

    fluds_shape = (self.n_angles, self.mesh.n_cells,
                   2*self.mesh.dim, self.n_groups)
    self._fluds = np.zeros(fluds_shape)

    # Initialize precursors
    if self.use_precursors:
        precursor_shape = (self.max_precursors, n_nodes)
        self.precursors = np.zeros(precursor_shape)
