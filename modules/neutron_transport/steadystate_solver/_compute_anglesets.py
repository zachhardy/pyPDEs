import numpy as np
from numpy import ndarray
from typing import List
from pyPDEs.utilities import Vector

from ..directed_graph import DirectedGraph

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


def initialize_angle_sets(self: 'SteadyStateSolver') -> None:
    """
    Initialize the angle sets for the problem.
    """
    self.angle_sets.clear()

    # Octant aggregation
    if self.angle_aggregation_type == 'octant':

        # 1D hemisphere aggregation
        if self.mesh.dim == 1:
            # Create angle sets
            top, bot = AngleSet(), AngleSet()

            # Loop over angles
            for i, omega in enumerate(self.quadrature.omegas):

                # Top hemisphere
                if omega.z > 0.0:
                    if not top.angles:
                        sweep_order = self.create_sweep_ordering(omega)
                        top.sweep_ordering = sweep_order
                    top.angles.append(i)

                # Bottom hemisphere
                if omega.z < 0.0:
                    if not bot.angles:
                        sweep_order = self.create_sweep_ordering(omega)
                        bot.sweep_ordering = sweep_order
                    bot.angles.append(i)

            # Add angle sets to solver
            self.angle_sets = [top, bot]

        # 2D quadrant aggregation
        if self.mesh.dim == 2:
            # Create angle sets
            top_right, top_left = AngleSet(), AngleSet()
            bot_left, bot_right = AngleSet(), AngleSet()

            # Loop over directions
            for i, omega in enumerate(self.quadrature.omegas):

                # Top right
                if omega.x > 0.0 and omega.y > 0.0:
                    if not top_right.angles:
                        sweep_order = self.create_sweep_ordering(omega)
                        top_right.sweep_ordering = sweep_order
                    top_right.angles.append(i)

                # Top left
                if omega.x < 0.0 and omega.y > 0.0:
                    if not top_left.angles:
                        sweep_order = self.create_sweep_ordering(omega)
                        top_left.sweep_ordering = sweep_order
                    top_left.angles.append(i)

                # Bottom left
                if omega.x < 0.0 and omega.y < 0.0:
                    if not bot_left.angles:
                        sweep_order = self.create_sweep_ordering(omega)
                        bot_left.sweep_ordering = sweep_order
                    bot_left.angles.append(i)

                # Bottom right
                if omega.x > 0.0 and omega.y < 0.0:
                    if not bot_right.angles:
                        sweep_order = self.create_sweep_ordering(omega)
                        bot_right.sweep_ordering = sweep_order
                    bot_right.angles.append(i)

            # Add angle sets to the solver
            self.angle_sets = [top_right, top_left, bot_left, bot_right]

def create_sweep_ordering(self: 'SteadyStateSolver',
                          omega: Vector) -> List[int]:
    """
    Create a sweep ordering in the direction omega.

    Parameters
    ----------
    omega : Vector

    Returns
    -------
    List[int]
    """
    # List of cell successors
    successors = [set() for _ in range(self.mesh.n_cells)]

    # Loop over cells
    for cell in self.mesh.cells:
        c = cell.id

        # Loop over faces
        for face in cell.faces:

            # Outgoing faces
            if omega.dot(face.normal) > 0.0:

                # If an interior face
                if face.has_neighbor:
                    successors[c].add(face.neighbor_id)

    # Create directed graph
    dg = DirectedGraph()

    # Add vertices
    for c in range(self.mesh.n_cells):
        dg.add_vertex()

    # Establish connections
    for c in range(self.mesh.n_cells):
        for successor in successors[c]:
            dg.add_edge(c, successor)

    return dg.create_topological_sort()

