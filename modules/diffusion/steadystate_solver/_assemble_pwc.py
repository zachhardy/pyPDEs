from scipy.sparse import csr_matrix
from numpy import ndarray
    
from pyPDEs.spatial_discretization import (PiecewiseContinuous,
                                           FiniteVolume)
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import (DirichletBoundary,
                                         NeumannBoundary,
                                         RobinBoundary)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SteadyStateSolver

def _pwc_assemble_diffusion_matrix(
        self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multi-group diffusion matrix.

    This routine does not include fission or scattering
    terms.

    Returns
    -------
    csr_matrix
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                # Loop over trial functions
                for j in range(view.n_nodes):
                    jg = pwc.map_dof(cell, j, uk_man, 0, g)

                    mass_ij = view.intV_shapeI_shapeJ[i][j]
                    stiff_ij = view.intV_gradI_gradJ[i][j]

                    # Diffusion + reaction term
                    value = xs.sigma_t[g] * mass_ij + \
                            xs.diffusion_coeff[g] * stiff_ij
                    rows.append(ig)
                    cols.append(jg)
                    data.append(value)

        # Loop over faces
        for f_id, face in enumerate(cell.faces):

            # Boundary terms
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)

                # Loop over groups
                for g in range(self.n_groups):
                    bc = self.boundaries[bndry_id * self.n_groups + g]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ig = pwc.map_face_dof(
                                cell, f_id, fi, uk_man, 0, g)

                            pwc.zero_dirichlet_row(ig, rows, data)
                            rows.append(ig)
                            cols.append(ig)
                            data.append(1.0)

                    # Robin boundary
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ig = pwc.map_face_dof(
                                cell, f_id, fi, uk_man, 0, g)

                            # Loop over face nodes
                            for fj in range(n_face_nodes):
                                nj = view.face_node_mapping[f_id][fj]
                                jg = pwc.map_face_dof(
                                    cell, f_id, fj, uk_man, 0, g)

                                face_mass_ij = \
                                    view.intS_shapeI_shapeJ[f_id][ni][nj]

                                value = bc.a / bc.b * face_mass_ij
                                rows.append(ig)
                                cols.append(jg)
                                data.append(value)

    n_dofs = pwc.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _pwc_assemble_scattering_matrix(
        self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multi-group scattering matrix.

    Returns
    -------
    csr_matrix
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                # Loop over trial functions
                for j in range(view.n_nodes):
                    mass_ij = view.intV_shapeI_shapeJ[i][j]

                    # Loop over groups
                    for gp in range(self.n_groups):
                        jgp = pwc.map_dof(cell, j, uk_man, 0, gp)

                        value = xs.sigma_tr[gp][g] * mass_ij
                        rows.append(ig)
                        cols.append(jgp)
                        data.append(value)

        # Loop over faces
        for f_id, face in enumerate(cell.faces):

            # Boundary terms
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)

                # Loop over groups
                for g in range(self.n_groups):
                    bc = self.boundaries[bndry_id * self.n_groups + g]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ig = pwc.map_face_dof(
                                cell, f_id, fi, uk_man, 0, g)
                            pwc.zero_dirichlet_row(ig, rows, data)

    n_dofs = pwc.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _pwc_assemble_fission_matrix(
        self: "SteadyStateSolver") -> csr_matrix:
    """Assemble the multi-group scattering matrix.

    Returns
    -------
    csr_matrix
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                # Loop over trial functions
                for j in range(view.n_nodes):
                    mass_ij = view.intV_shapeI_shapeJ[i][j]

                    # Loop over groups
                    for gp in range(self.n_groups):
                        jgp = pwc.map_dof(cell, j, uk_man, 0, gp)

                        # Total fission
                        if not self.use_precursors:
                            coeff = xs.chi[g] * xs.nu_sigma_f[gp]
                        else:
                            # Prompt fission
                            coeff = xs.chi_prompt[g] * \
                                    xs.nu_prompt_sigma_f[gp]

                            # Delayed fission
                            for p in range(xs.n_precursors):
                                coeff += xs.chi_delayed[g][p] * \
                                         xs.precursor_yield[p] * \
                                         xs.nu_delayed_sigma_f[gp]

                        value = coeff * mass_ij
                        rows.append(ig)
                        cols.append(jgp)
                        data.append(value)

        # Loop over faces
        for f_id, face in enumerate(cell.faces):

            # Boundary terms
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)

                # Loop over groups
                for g in range(self.n_groups):
                    bc = self.boundaries[bndry_id * self.n_groups + g]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ig = pwc.map_face_dof(
                                cell, f_id, fi, uk_man, 0, g)
                            pwc.zero_dirichlet_row(ig, rows, data)

    n_dofs = pwc.n_dofs(uk_man)
    return csr_matrix((data, (rows, cols)), shape=(n_dofs,) * 2)


def _pwc_set_source(self: "SteadyStateSolver",
                  apply_material_source: bool = True,
                  apply_boundary_source: bool = True,
                  apply_scattering_source: bool = False,
                  apply_fission_source: bool = False) -> None:
    """Assemble the right-hand side.

    This routine assembles the material source, scattering source,
    fission source, and boundary source based upon the provided flags.

    Parameters
    ----------
    apply_material_source : bool, default True
    apply_boundary_source : bool, default True
    apply_scattering_source : bool, default False
    apply_fission_source : bool, default False
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # Loop over cells
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]
        src = self.material_src[cell.material_id]

        # Loop over test functions
        for i in range(view.n_nodes):
            intV_shapeI = view.intV_shapeI[i]

            # Loop over groups
            for g in range(self.n_groups):
                ig = pwc.map_dof(cell, i, uk_man, 0, g)

                # Material source
                if apply_material_source:
                    self.b[ig] += src.values[g] * intV_shapeI

                # Loop over trial functions
                for j in range(view.n_nodes):
                    mass_ij = view.intV_shapeI_shapeJ[i][j]

                    # Loop over groups
                    for gp in range(self.n_groups):
                        jgp = pwc.map_dof(cell, j, uk_man, 0, gp)

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

                        self.b[ig] += coeff * self.phi[jgp] * mass_ij

        # Loop over faces
        for f_id, face in enumerate(cell.faces):

            # Boundary conditions
            if not face.has_neighbor and apply_boundary_source:
                bndry_id = -1 * (face.neighbor_id + 1)

                # Loop over groups
                for g in range(self.n_groups):
                    bc = self.boundaries[bndry_id * self.n_groups + g]

                    # Dirichlet boundary
                    if issubclass(type(bc), DirichletBoundary):
                        bc: DirichletBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ig = pwc.map_face_dof(cell, f_id, fi, uk_man, 0, g)
                            self.b[ig] = bc.value

                    # Neumann boundary
                    elif issubclass(type(bc), NeumannBoundary):
                        bc: NeumannBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ig = pwc.map_face_dof(cell, f_id, fi, uk_man, 0, g)

                            intS_shapeI = view.intS_shapeI[f_id][ni]
                            self.b[ig] += bc.value * intS_shapeI

                    # Robin boundary
                    elif issubclass(type(bc), RobinBoundary):
                        bc: RobinBoundary = bc

                        # Loop over face nodes
                        n_face_nodes = len(view.face_node_mapping[f_id])
                        for fi in range(n_face_nodes):
                            ni = view.face_node_mapping[f_id][fi]
                            ig = pwc.map_face_dof(cell, f_id, fi, uk_man, 0, g)

                            intS_shapeI = view.intS_shapeI[f_id][ni]
                            self.b[ig] += bc.f / bc.b * intS_shapeI


def _pwc_compute_precursors(self: "SteadyStateSolver") -> None:
    """Compute the delayed neutron precursor concentration."""
    pwc: PiecewiseContinuous = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man

    # Loop over cells
    self.precursors *= 0.0
    for cell in self.mesh.cells:
        volume = cell.volume
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # Loop over precursors
        for p in range(xs.n_precursors):
            dof = cell.id * prec_uk_man.total_components + p
            coeff =  xs.precursor_yield[p] / xs.precursor_lambda[p]

            # Loop over nodes
            for i in range(view.n_nodes):
                intV_shapeI = view.intV_shapeI[i]

                # Loop over groups
                for g in range(self.n_groups):
                    ig = pwc.map_dof(cell, i, flux_uk_man, 0, g)
                    self.precursors[dof] += \
                        coeff * xs.nu_delayed_sigma_f[g] * \
                        self.phi[ig] * intV_shapeI / cell.volume
