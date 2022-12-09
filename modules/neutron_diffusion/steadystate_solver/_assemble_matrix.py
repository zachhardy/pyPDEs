from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

import numpy as np
from scipy.sparse import csr_matrix

from ..boundaries import RobinBoundary


def _assemble_matrix(
        self: 'SteadyStateSolver',
        with_scattering: bool = False,
        with_fission: bool = False
) -> None:
    """
    Assemble the multi-group matrix.

    By default, this routine constructs the full multi-group operator.
    Optionally, one can omit the scattering and/or fission term. If both
    are omitted, the result is a symmetric positive definite matrix that
    is uncoupled in energy group. Solutions to this system require an
    iterative procedure.

    Parameters
    ----------
    self : SteadyStateSolver
    with_scattering : bool
    with_fission : bool
    """

    # Data for sparse matrix construction
    rows, cols, data = [], [], []

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        uk_map = self.n_groups * cell.id

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # Get material properties
        sig_t = self.cellwise_xs[cell.id].sigma_t
        D, B = xs.diffusion_coeff, xs.buckling

        # Group-to-group cell matrix
        Aloc = np.zeros((self.n_groups, self.n_groups))

        # Loop over groups
        for g in range(self.n_groups):

            # ========================================
            # Total interaction term + buckling
            # ========================================

            Aloc[g][g] += sig_t[g] + D[g] * B[g]

            # ========================================
            # Scattering term
            # ========================================

            if with_scattering:
                sig_s = xs.transfer_matrices[0][g]
                for gp in range(self.n_groups):
                    Aloc[g][gp] -= sig_s[gp]

            # ========================================
            # Fission term
            # ========================================

            if with_fission and xs.is_fissile:

                # Total fission
                if not self.use_precursors:
                    chi = xs.chi[g]
                    nu_sigf = xs.nu_sigma_f
                    for gp in range(self.n_groups):
                        Aloc[g][gp] -= chi * nu_sigf[gp]

                # Prompt + delayed fission
                else:
                    chi_p = xs.chi_prompt[g]
                    chi_d = xs.chi_delayed[g]
                    nup_sigf = xs.nu_prompt_sigma_f
                    nud_sigf = xs.nu_delayed_sigma_f
                    gamma = xs.precursor_yield
                    for gp in range(self.n_groups):
                        f = chi_p * nup_sigf[gp]
                        for j in range(xs.n_precursors):
                            f += chi_d[j] * gamma[j] * nud_sigf[j]
                        Aloc[g][gp] -= f

        # Add to matrix
        for g in range(self.n_groups):
            for gp in range(self.n_groups):
                rows.append(uk_map + g)
                cols.append(uk_map + gp)
                data.append(Aloc[g][gp] * volume)

        # ========================================
        # Diffusion and boundary terms
        # ========================================

        for face in cell.faces:

            # Interior diffusion
            if face.has_neighbor:
                nbr_cell = self.mesh.cells[face.neighbor_id]
                nbr_uk_map = nbr_cell.id * self.n_groups

                nbr_xs_id = self.matid_to_xs_map[nbr_cell.material_id]
                nbr_xs = self.material_xs[nbr_xs_id]
                D_nbr = nbr_xs.diffusion_coeff

                d_pn = (cell.centroid - nbr_cell.centroid).norm()
                d_pf = (cell.centroid - face.centroid).norm()
                w = d_pf / d_pn

                # Loop over groups
                for g in range(self.n_groups):
                    D_f = 1.0 / (w / D[g] + (1.0 - w) / D_nbr[g])
                    val = D_f / d_pn * face.area
                    rows.extend([uk_map + g, uk_map + g])
                    cols.extend([uk_map + g, nbr_uk_map + g])
                    data.extend([val, -val])

            # Boundary terms
            else:
                bid = face.neighbor_id
                btype = self.boundary_info[bid][0]

                # ========================================
                # Dirichlet boundary term
                # ========================================

                if btype == "ZERO_FLUX" or btype == "DIRICHLET":
                    d_pf = (cell.centroid - face.centroid).norm()
                    for g in range(self.n_groups):
                        val = D[g] / d_pf * face.area

                        rows.append(uk_map + g)
                        cols.append(uk_map + g)
                        data.append(val)

                elif (btype == "VACUUM" or
                      btype == "MARSHAK" or
                      btype == "ROBIN"):
                    d_pf = (cell.centroid - face.centroid).norm()
                    for g in range(self.n_groups):
                        bc: RobinBoundary = self.boundaries[bid][g]
                        val = bc.a * D[g] / (bc.b * D[g] + bc.a * d_pf)

                        rows.append(uk_map + g)
                        cols.append(uk_map + g)
                        data.append(val * face.area)

    # Construct the sparse matrix
    self._A = [csr_matrix((data, (rows, cols)))]
