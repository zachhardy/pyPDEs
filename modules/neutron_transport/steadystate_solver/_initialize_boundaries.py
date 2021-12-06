import numpy as np
from numpy import ndarray

from pyPDEs.utilities import Vector
from pyPDEs.material import *
from ..boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver


def initialize_bondaries(self: 'SteadyStateSolver') -> None:
    """
    Initialize the boundary conditions for the simulation.
    """
    ihat = Vector(x=1.0)
    jhat = Vector(y=1.0)
    khat = Vector(z=1.0)

    # Loop over boundary conditions
    for b, bc in enumerate(self.boundaries):
        if isinstance(bc, VacuumBoundary):
            bc.values = [0.0] * self.n_groups
        elif isinstance(bc, HomogeneousBoudary):
            if len(bc.values) != self.n_groups:
                raise ValueError(
                    f'Incident homogeneous boundary conditions must '
                    f'have as many values as the number of groups.')
        elif isinstance(bc, ReflectiveBoundary):
            if self.mesh.dim == 1:
                if b == 0: bc.normal = -khat
                if b == 1: bc.normal = khat
            else:
                if b == 0: bc.normal = -ihat
                if b == 1: bc.normal = ihat
                if b == 2: bc.normal = -jhat
                if b == 3: bc.normal = jhat
                if b == 4: bc.normal = -khat
                if b == 5: bc.normal = khat
            self._initialize_reflective_bc(bc)


def _initialize_reflective_bc(self: 'SteadyStateSolver',
                              bc: Boundary) -> None:
    """
    Initialize the structures for a reflective boundary condition.

    Parameters
    ----------
    bc : Boundary
    """
    if not isinstance(bc, ReflectiveBoundary):
        raise TypeError('The specified BC is not reflective.')

    # Compute reflected angles
    bc.reflected_angles = [-1 for _ in range(self.n_angles)]
    for n in range(self.n_angles):
        omega = self.quadrature.omegas[n]
        omega_refl = omega - 2.0*bc.normal*omega.dot(bc.normal)

        for ns in range(self.n_angles):
            omega_ns = self.quadrature.omegas[ns]
            if omega_refl.dot(omega_ns) > 1.0 - 1.0e-8:
                bc.reflected_angles[n] = ns
                break

        # Check that reflected angle exists
        if bc.reflected_angles[n] < 0:
            raise AssertionError(
                f'Reflected angle not found for angle {n} with direction '
                f'({omega.x}, {omega.y}, {omega.z}). Ensure that the '
                f'quadrature set is symmetric.')

    # Initialize boundary psi
    bc.boundary_psi = [[] for _ in range(self.n_angles)]
    for n in range(self.n_angles):
        bc.boundary_psi.append([])

        # Skip incident angles
        if self.quadrature.omegas[n].dot(bc.normal) < 0.0:
            continue

        # Loop over cells
        cell_vec = [[] for _ in range(self.mesh.n_cells)]
        for cell in self.mesh.cells:
            c = cell.id

            # Skip cells not on boundary
            on_bndry = False
            for face in cell.faces:
                if not face.has_neighbor:
                    if face.normal.dot(bc.normal) > 1.0 - 1.0e-8:
                        on_bndry = True
                        break
            if not on_bndry:
                continue

            # Loop over boundary faces
            cell_vec[c] = [[] for _ in range(len(cell.faces))]
            for f in range(len(cell.faces)):
                face = cell.faces[f]
                if not face.has_neighbor:
                    if face.normal.dot(bc.normal) > 1.0 - 1.0e-8:
                        cell_vec[c][f] = [0.0 for _ in range(self.n_groups)]
        bc.boundary_psi[n] = cell_vec
