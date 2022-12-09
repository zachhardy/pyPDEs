from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver

from ..boundaries import DirichletBoundary
from ..boundaries import NeumannBoundary
from ..boundaries import RobinBoundary


def _assemble_transient_rhs(
        self: 'TransientSolver',
        with_material_src: bool = False,
        with_boundary_src: bool = False,
        with_scattering: bool = False,
        with_fission: bool = False,
        step=0
) -> None:
    """
    Assemble the transient right-hand side vector for multi-group system.

    By default, this routine only contributes the material source
    term and boundary sources to the right-hand side vector. The
    four source options can be freely toggled on and off. Additionally,
    for multistep methods, the step parameter is used to define the step
    of the method.

    Notes
    -----
    The scattering and fission flags should be set opposite that
    of the flags used to construct the matrix. In other words, when
    scattering and fission terms are included in the matrix, they should
    not be included in the right-hand side and vice verse.

    This routine is additive, so the right-hand side must be cleared
    before calling the method, if necessary.

    Parameters
    ----------
    self : steadystate_solver
    with_material_src : bool
    with_scattering : bool
    with_fission : bool
    """
    eff_dt = self.effective_dt(step)
    t = self.time + eff_dt

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        uk_map_g = self.n_groups * cell.id
        uk_map_j = self.max_precursors * cell.id

        xs_id = self.matid_to_xs_map[cell.material_id]
        xs = self.material_xs[xs_id]

        src = None
        src_id = self.matid_to_src_map[cell.material_id]
        if with_material_src and src_id >= 0:
            src = self.material_src[src_id].values

        # Loop over groups
        for g in range(self.n_groups):
            rhs = 0.0

            # ========================================
            # Material source term
            # ========================================

            rhs += 0.0 if src is None else src[g]

            # ========================================
            # Old scalar flux
            # ========================================

            phi_old = self.phi_old[uk_map_g + g]
            if step == 1 and self.time_stepping_method == "TBDF2":
                phi = self.phi[uk_map_g + g]
                phi_old = (4.0 * phi - phi_old) / 3.0
            rhs += xs.inv_velocity[g] / eff_dt * phi_old

            # ========================================
            # Old precursors
            # ========================================

            if xs.is_fissile and self.use_precursors:
                chi_d = xs.chi_delayed[g]
                decay = xs.precursor_lambda
                for j in range(xs.n_precursors):
                    coeff = chi_d[j] * decay[j]
                    if not self.lag_precursors:
                        coeff /= 1.0 + eff_dt * decay[j]

                    c_old = self.precursors_old[uk_map_j + j]
                    if step == 1 and not self.lag_precursors:
                        c = self.precursors[uk_map_j + j]
                        c_old = (4.0 * c - c_old) / 3.0

                    rhs += coeff * c_old

            # ========================================
            # Scattering source term
            # ========================================

            if with_scattering:
                sig_s = xs.transfer_matrices[0][g]
                for gp in range(self.n_groups):
                    rhs += sig_s[gp] * self.phi[uk_map_g + gp]

            # ========================================
            # Fission source term
            # ========================================

            if with_fission and xs.is_fissile:

                # Total fission
                if not self.use_precursors:
                    chi = xs.chi[g]
                    nu_sigf = xs.nu_sigma_f
                    for gp in range(self.n_groups):
                        rhs += (chi * nu_sigf[gp] *
                                self.phi[uk_map_g + gp])

                # Prompt + delayed fission
                else:
                    # Prompt
                    chi_p = xs.chi_prompt[g]
                    nup_sigf = xs.nu_prompt_sigma_f
                    for gp in range(self.n_groups):
                        rhs += chi_p * nup_sigf[gp] * self.phi[uk_map_g + gp]

                    # Delayed
                    if not self.lag_precursors:
                        chi_d = xs.chi_delayed[g]
                        nud_sigf = xs.nu_delayed_sigma_f
                        decay = xs.precursor_lambda
                        gamma = xs.precursor_yield

                        coeff = 0.0
                        for j in range(xs.n_precursors):
                            coeff += (chi_d[j] * decay[j] * gamma[j] *
                                      eff_dt / (1.0 + eff_dt * decay[j]))

                        for gp in range(self.n_groups):
                            rhs += (coeff * nud_sigf[gp] *
                                    self.phi[uk_map_g + gp])

            self._b[uk_map_g + g] += rhs * volume

        # ========================================
        # Boundary source terms
        # ========================================

        if with_boundary_src:

            # Loop over faces
            for face in cell.faces:

                # Skip interior faces
                if face.has_neighbor:
                    continue

                bid = face.neighbor_id
                btype = self.boundary_info[bid][0]

                # ========================================
                # Dirichlet boundary source term
                # ========================================

                if btype == "DIRICHLET":
                    D = xs.diffusion_coeff
                    d_pf = cell.centroid.distance(face.centroid)
                    for g in range(self.n_groups):
                        bc: DirichletBoundary = self.boundaries[bid][g]
                        bc_val = bc.boundary_value(face.centroid, t)

                        self._b[uk_map_g + g] += \
                            D[g] / d_pf * bc_val * face.area

                # ========================================
                # Neumann boundary source term
                # ========================================

                elif btype == "NEUMANN":
                    for g in range(self.n_groups):
                        bc: NeumannBoundary = self.boundaries[bid][g]
                        bc_val = bc.boundary_value(face.centroid, t)

                        self._b[uk_map_g + g] += bc_val * face.area

                # ========================================
                # Robin boundary source term
                # ========================================

                elif btype == "MARSHAK" or btype == "ROBIN":
                    D = xs.diffusion_coeff
                    d_pf = (cell.centroid - face.centroid).norm()
                    for g in range(self.n_groups):
                        bc: RobinBoundary = self.boundaries[bid][g]
                        bc_val = bc.boundary_value(face.centroid, t)

                        coeff = D[g] / (bc.b * D[g] + bc.a * d_pf)
                        self._b[uk_map_g + g] += coeff * bc_val * face.area
