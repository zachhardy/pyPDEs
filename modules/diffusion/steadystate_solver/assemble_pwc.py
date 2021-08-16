from scipy.sparse import csr_matrix
from numpy import ndarray
from typing import TYPE_CHECKING

from pyPDEs.spatial_discretization import (PiecewiseContinuous,
                                           FiniteVolume)
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import (DirichletBoundary,
                                         NeumannBoundary,
                                         RobinBoundary)

if TYPE_CHECKING:
    from .steadystate_solver import SteadyStateSolver


def pwc_assemble_matrix(self: 'SteadyStateSolver', g: int) -> csr_matrix:
    """
    Assemble the diffusion matrix for group `g`.

    Parameters
    ----------
    g : int
        The energy group under consideration.

    Returns
    -------
    csr_matrix
        The diffusion matrix for group `g`.
    """
    pwc: PiecewiseContinuous = self.discretization

    # ======================================== Loop over cells
    rows, cols, data = [], [], []
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # =================================== Loop over test functions
        for i in range(view.n_nodes):
            ii = pwc.map_dof(cell, i)

            # ============================== Loop over trial functions
            for k in range(view.n_nodes):
                kk = pwc.map_dof(cell, k)
                mass_ik = view.intV_shapeI_shapeJ[i][k]
                stiff_ik = view.intV_gradI_gradJ[i][k]

                # ==================== Reaction + diffusion term
                value = xs.sigma_t[g] * mass_ik
                value += xs.diffusion_coeff[g] * stiff_ik
                rows.append(ii)
                cols.append(kk)
                data.append(value)

        # ======================================== Loop over faces
        #                                          stop on boundaries
        for f_id, face in enumerate(cell.faces):
            if not face.has_neighbor:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id * self.n_groups + g]

                # ========================= Dirichlet boundary
                if issubclass(type(bc), DirichletBoundary):

                    # ==================== Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f_id])
                    for fi in range(n_face_nodes):
                        ii = pwc.map_face_dof(cell, f_id, fi)
                        pwc.zero_dirichlet_row(ii, rows, data)
                        rows.append(ii)
                        cols.append(ii)
                        data.append(1.0)

                # ============================== Robin boundary
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc

                    # ==================== Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f_id])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f_id][fi]
                        ii = pwc.map_face_dof(cell, f_id, fi)

                        # =============== Loop over face nodes
                        for fj in range(n_face_nodes):
                            nj = view.face_node_mapping[f_id][fj]
                            jj = pwc.map_face_dof(cell, f_id, fj)

                            value = bc.a / bc.b
                            value *= view.intS_shapeI_shapeJ[f_id][ni][nj]
                            rows.append(ii)
                            cols.append(jj)
                            data.append(value)

    return csr_matrix((data, (rows, cols)), shape=(pwc.n_nodes,) * 2)


def pwc_set_source(self: 'SteadyStateSolver', g: int, phi: ndarray,
                   apply_material_source: bool = True,
                   apply_scattering: bool = True,
                   apply_fission: bool = True,
                   apply_boundaries: bool = True) -> None:
    """
    Assemble the right-hand side of the diffusion equation.
    This includes material, scattering, fission, and boundary
    sources for group `g`.

    Parameters
    ----------
    g : int
        The group under consideration
    phi : ndarray
        A vector to compute scattering and fission sources with.
    apply_material_source : bool, default True
    apply_scattering : bool, default True
    apply_fission : bool, default True
    apply_boundaries : bool, default True
    """
    pwc: PiecewiseContinuous = self.discretization
    uk_man: UnknownManager = self.flux_uk_man

    # ============================================= Loop over cells
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]
        src = self.material_src[cell.material_id].values[g]

        # ======================================== Loop over test functions
        for i in range(view.n_nodes):
            ig = pwc.map_dof(cell, i, uk_man, 0, g)

            # ==================== Material source
            if apply_material_source:
                self.b[ig] += src * view.intV_shapeI[i]

            # =================================== Loop over trial fucntions
            for k in range(view.n_nodes):
                mass_ik = view.intV_shapeI_shapeJ[i][k]

                # ============================== Loop over groups
                for gp in range(self.n_groups):
                    kgp = pwc.map_dof(cell, k, uk_man, 0, gp)

                    # ==================== Scattering source
                    if apply_scattering:
                        self.b[ig] += \
                            xs.sigma_tr[gp][g] * phi[kgp] * mass_ik

                    # ==================== Fission source
                    if apply_fission:
                        # Without delayed neutrons
                        if not self.use_precursors:
                            self.b[ig] += xs.chi[g] * \
                                          xs.nu_sigma_f[gp] * \
                                          phi[kgp] * mass_ik

                        # With delayed neutrons
                        else:
                            # =============== Prompt fission
                            self.b[ig] += xs.chi_prompt[g] * \
                                          xs.nu_prompt_sigma_f[gp] * \
                                          phi[kgp] * mass_ik

                            # =============== Delayed fission
                            for j in range(xs.n_precursors):
                                self.b[ig] += xs.chi_delayed[g][j] * \
                                              xs.precursor_yield[j] * \
                                              xs.nu_delayed_sigma_f[gp] * \
                                              phi[kgp] * mass_ik

        # ======================================== Loop over faces
        #                                          Stop on boundaries
        for f_id, face in enumerate(cell.faces):
            if not face.has_neighbor and apply_boundaries:
                bndry_id = -1 * (face.neighbor_id + 1)
                bc = self.boundaries[bndry_id * self.n_groups + g]

                # ============================== Dirichlet boundary
                if issubclass(type(bc), DirichletBoundary):
                    bc: DirichletBoundary = bc

                    # ==================== Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f_id])
                    for fi in range(n_face_nodes):
                        ig = pwc.map_face_dof(cell, f_id, fi, uk_man, 0, g)
                        self.b[ig] = bc.value

                # ============================== Neumann boundary
                elif issubclass(type(bc), NeumannBoundary):
                    bc: NeumannBoundary = bc

                    # ==================== Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f_id])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f_id][fi]
                        ig = pwc.map_face_dof(cell, f_id, fi, uk_man, 0, g)
                        self.b[ig] += bc.value * view.intS_shapeI[f_id][ni]

                # ============================== Robin boundary
                elif issubclass(type(bc), RobinBoundary):
                    bc: RobinBoundary = bc

                    # ==================== Loop over face nodes
                    n_face_nodes = len(view.face_node_mapping[f_id])
                    for fi in range(n_face_nodes):
                        ni = view.face_node_mapping[f_id][fi]
                        ig = pwc.map_face_dof(cell, f_id, fi, uk_man, 0, g)
                        self.b[ig] += bc.f / bc.b * view.intS_shapeI[f_id][ni]


def pwc_compute_precursors(self: 'SteadyStateSolver') -> None:
    """
    Compute the delayed neutron precursor concentration.
    """
    pwc: PiecewiseContinuous = self.discretization
    flux_uk_man = self.flux_uk_man
    prec_uk_man = self.precursor_uk_man
    self.precursors *= 0.0

    # ======================================== Loop over cells
    for cell in self.mesh.cells:
        view = pwc.fe_views[cell.id]
        xs = self.material_xs[cell.material_id]

        # =================================== Loop over precursors
        for j in range(xs.n_precursors):
            ij = cell.id * prec_uk_man.total_components + j
            coeff = \
                xs.precursor_yield[j] / xs.precursor_lambda[j]

            # ============================== Loop over nodes
            for i in range(view.n_nodes):
                intV_shapeI = view.intV_shapeI[i]

                # ========================= Loop over groups
                for g in range(self.n_groups):
                    ig = pwc.map_dof(cell, i, flux_uk_man, 0, g)
                    self.precursors[ij] += \
                        coeff * xs.nu_delayed_sigma_f[g] * \
                        self.phi[ig] * intV_shapeI / cell.volume
