import numpy as np
from numpy import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


class AngleSet:
    """
    Implementation for an angle set.

    An angle set is defined as a collection of angles that
    share a sweep ordering.
    """
    def __init__(self) -> None:
        self.angles: List[int] = []
        self.sweep_ordering: List[int] = []


def create_angle_sets(self: 'SteadyStateSolver') -> None:
    """
    Create the angle sets.
    """
    # Clear current angle sets
    self.angle_sets.clear()

    # 1D problems
    if self.mesh.dim == 1:
        sweep_order = list(range(self.mesh.n_cells))

        # Rightward directions
        angle_set = AngleSet()
        angle_set.sweep_ordering = sweep_order
        for i in range(self.n_angles):
            omega = self.quadrature.omegas[i]
            if omega.z > 0.0:
                angle_set.angles.append(i)
        self.angle_sets.append(angle_set)

        # Leftward directions
        angle_set = AngleSet()
        angle_set.sweep_ordering = sweep_order[::-1]
        for i in range(self.n_angles):
            omega = self.quadrature.omegas[i]
            if omega.z < 0.0:
                angle_set.angles.append(i)
        self.angle_sets.append(angle_set)

