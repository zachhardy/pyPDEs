from scipy.sparse import csr_matrix
from numpy import ndarray

from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import (DirichletBoundary,
                                         NeumannBoundary,
                                         RobinBoundary)

if TYPE_CHECKING:
    from . import SteadyStateSolver


def _fv_assemble_diffusion_matrix(
        self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multi-group diffusion matrix.

    This routine does not include fission or scattering
    terms.

    Returns
    -------
    csr_matrix
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)

            # Reaction term
            value = xs.sigma_t[g] * volume
            rows.append(ig)
            cols.append(ig)
            data.append(value)

            # Loop over faces
            for face in cell.faces:
                # Interior faces
                if face.has_neighbor:
                    nbr_cell = self.mesh.cells[face.neighbor_id]
                    nbr_xs = self.material_xs[nbr_cell.material_id]
                    jg = fv.map_dof(nbr_cell, 0, uk_man, 0, g)

                    # Diffusion coeffients
                    D_p = xs.diffusion_coeff[g]
                    D_n = nbr_xs.diffusion_coeff[g]

                    # Node-to-neighbor/face information
                    d_pn = (cell.centroid - nbr_cell.centroid).norm()
                    d_pf = (cell.centroid - face.centroid).norm()

                    # Face diffusion coefficient
                    w = d_pf / d_pn  # harmonic mean weight
                    D_f = (w / D_p + (1.0 - w) / D_n) ** (-1)

                    # Diffusion term
                    value = D_f / d_pn * face.area
                    rows.extend([ig, ig])
                    cols.extend([ig, jg])
                    data.extend([value, -value])


                # Boundary faces
                else:
                    bndry_id = -1 * (face.neighbor_id + 1)
                    bc = self.boundaries[bndry_id * self.n_groups + g]

                    D_p = xs.diffusion_coeff[g]
                    d_pf = (cell.centroid - face.centroid).norm()

                    # ==================== Boundary conditions
                    value = 0.0
                    if issubclass(type(bc), DirichletBoundary):
                        value = D_p / d_pf * face.area
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc
                        tmp = bc.a * d_pf - bc.b * D_p
                        value = bc.a * D_p / tmp * face.area

                    rows.append(ig)
                    cols.append(ig)
                    data.append(value)

    n_dofs = fv.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _fv_assemble_scattering_matrix(
        self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multi-group scattering matrix.

    Returns
    -------
    csr_matrix
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                value = xs.sigma_tr[gp][g] * volume
                rows.append(ig)
                cols.append(igp)
                data.append(value)

    n_dofs = fv.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _fv_assemble_fission_matrix(
        self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multi-group fission matrix.

    Returns
    -------
    csr_matrix
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                # Total fission
                if not self.use_precursors:
                    value = xs.chi[g] * xs.nu_sigma_f[gp]
                else:
                    # Prompt fission
                    value = xs.chi_prompt[g] * \
                            xs.nu_prompt_sigma_f[gp]

                    # Delayed fission
                    for j in range(xs.n_precursors):
                        value += xs.chi_delayed[g][j] * \
                                 xs.precursor_yield[j] * \
                                 xs.nu_delayed_sigma_f[gp]

                value *= volume
                rows.append(ig)
                cols.append(igp)
                data.append(value)

    n_dofs = fv.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _fv_set_source(self: "SteadyStateSolver",
                  apply_material_source: bool = True,
                  apply_boundary_source: bool = True,
                  apply_scattering_source: bool = True,
                  apply_fission_source: bool = True) -> None:
    """Assemble the right-hand side for group `g`.

    This routine assembles the material source, scattering source,
    fission source, and boundary source based upon the provided flags.

    Parameters
    ----------
    g : int
        The group under consideration
    apply_material_source : bool, default True
    apply_boundary_source : bool, default True
    apply_scattering_source : bool, default True
    apply_fission_source : bool, default True
    """
    fv: FiniteVolume = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    for cell in self.mesh.cells:
        volume = cell.volume
        xs = self.material_xs[cell.material_id]
        src = self.material_src[cell.material_id]

        # Loop over groups
        for g in range(self.n_groups):
            ig = fv.map_dof(cell, 0, uk_man, 0, g)

            # Material source
            if apply_material_source:
                self.b[ig] += src.values[g] * volume

            # Loop over groups
            for gp in range(self.n_groups):
                igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                coeff = 0.0  # total transfer coefficient

                # Scattering source
                if apply_scattering_source:
                    coeff += xs.sigma_tr[gp][g]

                # Fission source
                if apply_fission_source:
                    # Total fission
                    if not self.use_precursors:
                        coeff += xs.chi[g] * xs.nu_sigma_f[gp]
                    else:
                        # Prompt fission
                        coeff += xs.chi_prompt[g] * \
                                 xs.nu_prompt_sigma_f[gp]

                        # Delayed fission
                        for j in range(xs.n_precursors):
                            coeff += xs.chi_delayed[g][j] * \
                                     xs.precursor_yield[j] * \
                                     xs.nu_delayed_sigma_f[gp]

                self.b[ig] += coeff * self.phi[igp] * volume

        # Loop over faces
        for face in cell.faces:
            if not face.has_neighbor and apply_boundary_source:
                bndry_id = -1 * (face.neighbor_id + 1)

                # Distance from centroid to face
                d_pf = (cell.centroid - face.centroid).norm()

                # Loop over groups
                for g in range(self.n_groups):
                    bc = self.boundaries[bndry_id * self.n_groups + g]
                    ig = fv.map_dof(cell, 0, uk_man, 0, g)
                    D_p = xs.diffusion_coeff[g]

                    # Boundary conditions
                    value = 0.0
                    if issubclass(type(bc), DirichletBoundary):
                        bc: DirichletBoundary = bc
                        value = D_p / d_pf * bc.value
                    elif issubclass(type(bc), NeumannBoundary):
                        bc: NeumannBoundary = bc
                        value = bc.value
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc
                        tmp = bc.a * d_pf - bc.b * D_p
                        value = -bc.b * D_p / tmp * bc.f

                    self.b[ig] += value * face.area


def _fv_compute_precursors(self: "SteadyStateSolver") -> None:
    """Compute the delayed neutron precursor concentration."""
    fv: FiniteVolume = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man

    # Loop over cells
    self.precursors *= 0.0
    for cell in self.mesh.cells:
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for j in range(xs.n_precursors):
            ij = cell.id * prec_uk_man.total_components + j
            coeff = \
                xs.precursor_yield[j] / xs.precursor_lambda[j]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, flux_uk_man, 0, g)
                self.precursors[ij] += \
                    coeff * xs.nu_delayed_sigma_f[g] * self.phi[ig]
