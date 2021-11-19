import numpy as np
from numpy import ndarray

from pyPDEs.material import *
from ..boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def initialize_bondaries(self: 'SteadyStateSolver') -> None:
    """
    Initialize the boundary conditions for the simulation.
    """
    for b, bc in enumerate(self.boundaries):
        if isinstance(bc, VacuumBoundary):
            bc.values = [0.0] * self.n_groups
        elif isinstance(bc, IncidentHomogeneousBoundary):
            if len(bc.values) != self.n_groups:
                raise ValueError(
                    f'Incident homogeneous boundary conditions must '
                    f'have as many values as the number of groups.')
