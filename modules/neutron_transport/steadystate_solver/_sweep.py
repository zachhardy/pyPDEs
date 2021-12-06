import numpy as np

from pyPDEs.material import *
from pyPDEs.mesh import *
from pyPDEs.utilities import Vector
from ..data_structures import *
from ..boundaries import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

XS = CrossSections


def prepare_for_sweep(self: 'SteadyStateSolver') -> None:
    # Clear flux moments and angular fluxes
    self.phi *= 0.0
    self.psi *= 0.0

    # Reset angle ready flags
    for bc in self.boundaries:
        if isinstance(bc, ReflectiveBoundary):
            bc.reset_angle_ready_status()

    # Set executed status to Fasle
    for angle_set in self.angle_sets:
        angle_set.executed = False


def sweep(self: 'SteadyStateSolver'):
    """
    Perform a sweep over all angle sets.
    """
    self.prepare_for_sweep()

    sd = self.discretization
    A, b = 0.0, np.zeros(self.n_groups)

    # Go until all angle sets execute
    done_executed = False
    while not done_executed:

        # Assume finished, unless proven otherwise
        done_executed = True

        for angle_set_num in range(len(self.angle_sets)):
            angle_set: AngleSet = self.angle_sets[angle_set_num]

            # Skip if already executed
            if angle_set.executed:
                continue

            # Check if this angle set is ready. If no, skip and
            # set done_executed to false
            for bc in self.boundaries:
                if not bc.check_angle_ready_status(angle_set.angles):
                    done_executed = False
                    continue

            sweep_order = angle_set.sweep_ordering

            # Loop over sweep ordering
            n_cells = len(sweep_order)
            for so_index in range(n_cells):
                c = sweep_order[so_index]
                cell = self.mesh.cells[c]
                volume = cell.volume

                # Get material xs
                xs_id = self.matid_to_xs_map[cell.material_id]
                xs: XS = self.material_xs[xs_id]
                sig_t = xs.sigma_t

                # Loop over angles
                for angle_set_index in range(len(angle_set.angles)):
                    n = angle_set.angles[angle_set_index]
                    omega = self.quadrature.omegas[n]
                    w = self.quadrature.weights[n]

                    # Reset lhs and rhs
                    A *= 0.0
                    b *= 0.0

                    # Loop over faces
                    psi_inc = [Vector() for _ in range(self.n_groups)]
                    for f in range(len(cell.faces)):
                        face = cell.faces[f]
                        mu = omega.dot(face.normal)

                        # Upwind
                        if mu < 0.0:
                            if face.has_neighbor:
                                psi = self.psi_upwind(c, f, n)
                            else:
                                b_id = face.neighbor_id
                                psi = self.psi_boundary(b_id, c, f, n)

                            A -= 2.0 * mu * face.area
                            for g in range(self.n_groups):
                                b[g] -= 2.0*mu*face.area * psi[g]
                                psi_inc[g] += abs(face.normal * psi[g])

                    # Loop over groups
                    for g in range(self.n_groups):

                        # Loop over moments
                        for m in range(self.n_moments):
                            m2d = self.M[m][n]
                            b[g] += m2d * self.q_moments[m][g][c] * volume

                        # Solve for psi at cell center
                        psi_ijk = b[g] / (A + sig_t[g]*volume)

                        # Accumulate flux moment
                        for m in range(self.n_moments):
                            d2m = self.D[m][n]
                            self.phi[m][g][c] += d2m * psi_ijk

                        # Store angular flux
                        self.psi[n][g][c] = psi_ijk

                        # Store outgoing angular fluxes
                        for f in range(len(cell.faces)):
                            face: Face = cell.faces[f]

                            if omega.dot(face.normal) > 0.0:
                                psi_inc_dot_n = abs(face.normal.dot(psi_inc[g]))
                                psi_out = 2.0*psi_ijk - psi_inc_dot_n
                                psi_out = psi_out if psi_out > 0.0 else 0.0

                                # Interior faces
                                if face.has_neighbor:
                                    self.psi_outflow(psi_out, c, f, n, g)

                                # Reflecting boundaries
                                else:
                                    bndry_id = face.neighbor_id
                                    bc: Boundary = self.boundaries[bndry_id]
                                    if isinstance(bc, ReflectiveBoundary):
                                        bc.set_psi_outgoing(psi_out, c, f, n, g)

            # Declare this angle set executed
            angle_set.executed = True

            # Set angle ready flags on BCs
            for bc in self.boundaries:
                bc.set_angle_ready_status(angle_set.angles)