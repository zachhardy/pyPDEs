from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

from ..boundaries import DirichletBoundary
from ..boundaries import NeumannBoundary
from ..boundaries import RobinBoundary


def _assemble_rhs(
        self: 'SteadyStateSolver',
        with_material_src: bool = True,
        with_boundary_src: bool = True,
        with_scattering: bool = False,
        with_fission: bool = False
) -> None:
    """
    Assemble the right-hand side vector for multi-group system.

    By default, this routine only contributes the material source
    term and boundary sources to the right-hand side vector. The
    four source options can be freely toggled on and off.

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
    self : SteadyStateSolver
    with_material_src : bool
    with_scattering : bool
    with_fission : bool
    """

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        uk_map = self.n_groups * cell.id

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
            # Scattering source term
            # ========================================

            if with_scattering:
                sig_s = xs.transfer_matrices[0][g]
                for gp in range(self.n_groups):
                    rhs += sig_s[gp] * self.phi[uk_map + gp]

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
                                self.phi[uk_map + gp])

                # Prompt + delayed fission
                else:
                    chi_p = xs.chi_prompt[g]
                    chi_d = xs.chi_delayed[g]
                    nup_sigf = xs.nu_prompt_sigma_f
                    nud_sigf = xs.nu_delayed_sigma_f
                    gamma = xs.precursor_yield

                    for gp in range(self.n_groups):
                        rhs += (chi_p * nup_sigf[gp] *
                                self.phi[uk_map + gp])

                        for j in range(xs.n_precursors):
                            rhs += (chi_d[j] * gamma[j] *
                                    nud_sigf[gp] * self.phi[uk_map + gp])

            self._b[uk_map + g] += rhs * volume

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
                        self._b[uk_map + g] += \
                            D[g] / d_pf * bc.value * face.area

                # ========================================
                # Neumann boundary source term
                # ========================================

                elif btype == "NEUMANN":
                    for g in range(self.n_groups):
                        bc: NeumannBoundary = self.boundaries[bid][g]
                        self._b[uk_map + g] += bc.value * face.area

                # ========================================
                # Robin boundary source term
                # ========================================

                elif btype == "MARSHAK" or btype == "ROBIN":
                    D = xs.diffusion_coeff
                    d_pf = cell.centroid.distance(face.centroid)
                    for g in range(self.n_groups):
                        bc: RobinBoundary = self.boundaries[bid][g]
                        coeff = D[g] / (bc.b * D[g] + bc.a * d_pf)
                        self._b[uk_map + g] += coeff * bc.f * face.area
