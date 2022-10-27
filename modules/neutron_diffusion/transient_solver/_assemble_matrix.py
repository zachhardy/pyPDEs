from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver

import numpy as np
from scipy.sparse import csr_matrix

from ..boundaries import RobinBoundary


def _assemble_transient_matrices(
        self: 'TransientSolver',
        with_scattering: bool = False,
        with_fission: bool = False
) -> None:
    """
    Assemble the transient multi-group matrices.

    This is a convenience routine which constructs all necessary matrices
    for multistep methods and adds them into the list of matrices.

    Parameters
    ----------
    self : TransientSolver
    with_scattering : bool
    with_fission : bool
    """
    self._A = []
    self._assemble_transient_matrix(with_scattering, with_fission, 0)
    if self.time_stepping_method == "TBDF2":
        self._assemble_transient_matrix(with_scattering, with_fission, 1)


def _assemble_transient_matrix(
        self: 'TransientSolver',
        with_scattering: bool = True,
        with_fission: bool = True,
        step: int = 0
) -> None:
    """
    Assemble the transient multi-group matrix.

    By default, this routine constructs the full multi-group operator.
    Optionally, one can omit the scattering and/or fission term. If both
    are omitted, the result is a symmetric positive definite matrix that
    is uncoupled in energy group. Solutions to this system require an
    iterative procedure. Additionally, for multistep methods, the step
    parameter is used to define the step of the method.

    Parameters
    ----------
    self : TransientSolver
    with_scattering : bool
    with_fission : bool
    step : int
    """
    eff_dt = self.effective_dt(step)

    # Data for sparse matrix construction
    rows, cols, data = [], [], []

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        uk_map = self.n_groups * cell.id

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        # Dynamic cross-sections
        if xs.sigma_a_function is not None:
            t_update = self.time
            t_update += eff_dt if step == 0 else self.dt
            args = [t_update, self.temperature[cell.id]]
            self.cellwise_xs[cell.id].update(args)

        # Get material properties
        sig_t = self.cellwise_xs[cell.id].sigma_t
        D, B = xs.diffusion_coeff, xs.buckling
        inv_vel = xs.inv_velocity

        # Group-to-group cell matrix
        Aloc = np.zeros((self.n_groups, self.n_groups))

        # Loop over groups
        for g in range(self.n_groups):

            # ========================================
            # Total interaction term + buckling
            # ========================================

            Aloc[g][g] += sig_t[g] + D[g] * B[g]

            # ========================================
            # Time-derivative term
            # ========================================

            Aloc[g][g] += inv_vel[g] / eff_dt

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
                    # Prompt
                    chi_p = xs.chi_prompt[g]
                    nup_sigf = xs.nu_prompt_sigma_f
                    for gp in range(self.n_groups):
                        Aloc[g][gp] -= chi_p * nup_sigf[gp]

                    # Delayed
                    if not self.lag_precursors:
                        chi_d = xs.chi_delayed[g]
                        nud_sigf = xs.nu_delayed_sigma_f
                        decay = xs.precursor_lambda
                        gamma = xs.precursor_yield

                        # Precompute the scalar coefficient. This coefficient
                        # comes from the precursor substitution.
                        coeff = 0.0
                        for j in range(xs.n_precursors):
                            coeff += (chi_d[j] * decay[j] * gamma[j] *
                                      eff_dt / (1.0 + eff_dt * decay[j]))

                        # Delayed fission contribution
                        for gp in range(self.n_groups):
                            Aloc[g][gp] -= coeff * nud_sigf[gp]

        # Add to matrix
        for g in range(self.n_groups):
            for gp in range(self.n_groups):
                if Aloc[g][gp] != 0.0:
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

                d_pn = cell.centroid.distance(nbr_cell.centroid)
                d_pf = cell.centroid.distance(face.centroid)
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
                    d_pf = cell.centroid.distance(face.centroid)
                    for g in range(self.n_groups):
                        val = D[g] / d_pf * face.area

                        rows.append(uk_map + g)
                        cols.append(uk_map + g)
                        data.append(val)

                elif (btype == "VACUUM" or
                      btype == "MARSHAK" or
                      btype == "ROBIN"):
                    d_pf = cell.centroid.distance(face.centroid)
                    for g in range(self.n_groups):
                        bc: RobinBoundary = self.boundaries[bid][g]
                        val = (bc.a * D[g] /
                               (bc.b * D[g] + bc.a * d_pf) * face.area)

                        rows.append(uk_map + g)
                        cols.append(uk_map + g)
                        data.append(val)

    # Construct the sparse matrix
    self._A.append(csr_matrix((data, (rows, cols))))
