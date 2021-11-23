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


def sweep(self: 'SteadyStateSolver'):
    """
    Perform a sweep over all angle sets.
    """
    self.phi *= 0.0
    self.psi *= 0.0

    sd = self.discretization
    A, b = 0.0, np.zeros(self.n_groups)

    for angle_set in self.angle_sets:
        angle_set: AngleSet = angle_set

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

                A *= 0.0
                b *= 0.0

                # Loop over faces
                psi_inc_map, inc_to_outb_map = {}, {}
                for f in range(len(cell.faces)):
                    face = cell.faces[f]
                    mu = omega.dot(face.normal)

                    # Upwind
                    if mu < 0.0:
                        if face.has_neighbor:
                            psi = self.psi_upwind(c, f, n)
                        else:
                            bid = face.neighbor_id
                            psi = self.psi_boundary(bid, c, f, n)
                        psi_inc_map[f] = psi

                        # Find outbound face
                        for f_ in range(len(cell.faces)):
                            face_: Face = cell.faces[f_]
                            if face.normal == -face_.normal:
                                inc_to_outb_map[f] = f_

                        A -= 2.0 * mu * face.area
                        for g in range(self.n_groups):
                            b[g] -= 2.0*mu*face.area * psi[g]

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
                    for fi, fo in inc_to_outb_map.items():
                        face: Face = cell.faces[fo]

                        # Compute diamond difference relationship
                        psi_out = 2.0*psi_ijk - psi_inc_map[fi][g]

                        # Interior faces
                        if face.has_neighbor:
                            self.psi_outflow(psi_out, c, fo, n, g)

                        # Reflecting boundaries
                        else:
                            bndry_id = face.neighbor_id
                            bc: Boundary = self.boundaries[bndry_id]
                            if isinstance(bc, ReflectiveBoundary):
                                bc.set_psi_outgoing(psi_out, c, fo, n, g)
