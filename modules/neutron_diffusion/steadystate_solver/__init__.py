import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List

from pyPDEs.mesh import Mesh
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.spatial_discretization import SpatialDiscretization
from pyPDEs.spatial_discretization import FiniteVolume
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import *

class SteadyStateSolver:
    """Class for solving steady-state multigroup diffusion problems.
    """

    def __init__(self) -> None:
        """Class constructor.
        """
        # Domain objects
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = None

        # Materials information
        self.material_xs: List[CrossSections] = None
        self.material_src: List[MultiGroupSource] = None

        # Physics options
        self.use_precursors: bool = False

        # Precomputed matrices
        self.L: csr_matrix = None
        self.S: csr_matrix = None
        self.Fp: csr_matrix = None
        self.Fd: csr_matrix = None

        # Scalar flux solution vector
        self.phi: ndarray = None
        self.phi_uk_man: UnknownManager = UnknownManager()

        # Precursor solution vector
        self.precursors: ndarray = None
        self.precursor_uk_man: UnknownManager = UnknownManager()

    @property
    def n_groups(self) -> int:
        """Get the number of energy groups.

        Returns
        -------
        int
        """
        return self.material_xs[0].n_groups

    @property
    def n_precursors(self) -> int:
        """Get the total number of precursors.

        Returns
        -------
        int
        """
        n_precursors = 0
        for xs in self.material_xs:
            n_precursors += xs.n_precursors
        return n_precursors

    def initialize(self) -> None:
        """Initialize the solver.
        """
        self._check_inputs()
        sd = self.discretization

        # Initialize phi information
        self.phi_uk_man.clear()
        self.phi_uk_man.add_unknown(self.n_groups)
        self.phi = np.zeros(sd.n_dofs(self.phi_uk_man))

        # Initialize precursor information
        if self.use_precursors:
            # Compute the max precursors per material
            max_precursors: int = 0
            for xs in self.material_xs:
                if xs.n_precursors > max_precursors:
                    max_precursors = xs.n_precursors

            # Initialize vector and unknown manager
            self.precursor_uk_man.clear()
            if self.n_precursors > 0:
                self.precursor_uk_man.add_unknown(max_precursors)
                self.precursors = np.zeros(sd.n_dofs(self.precursor_uk_man))
            else:
                self.use_precursors = False

        # Precompute matrices
        self.L = self.diffusion_matrix()
        self.S = self.scattering_matrix()
        self.Fp = self.prompt_fission_matrix()
        self.Fd = self.delayed_fission_matrix()

    def execute(self) -> None:
        """Execute the steady-state multigroup diffusion solver.
        """
        A = self.L - self.S - self.Fp - self.Fd
        b = self.set_source()
        self.apply_vector_bcs(b)
        self.phi = spsolve(A, b)
        self.compute_precursors()

    def diffusion_matrix(self) -> csr_matrix:
        """Assemble the multigroup diffusion matrix.

        This routine assembles the diffusion plus interaction matrix
        for all groups according to the DoF ordering of `phi_uk_man`.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

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
                    if face.has_neighbor:  # interior faces
                        nbr_cell = self.mesh.cells[face.neighbor_id]
                        nbr_xs = self.material_xs[nbr_cell.material_id]
                        jg = fv.map_dof(nbr_cell, 0, uk_man, 0, g)

                        # Geometric information
                        d_pn = (cell.centroid - nbr_cell.centroid).norm()
                        d_pf = (cell.centroid - face.centroid).norm()
                        w = d_pf / d_pn

                        # Face diffusion coefficient
                        D_f = (w/xs.D[g] + (1.0 - w)/nbr_xs.D[g])**(-1)

                        # Diffusion term
                        value = D_f / d_pn * face.area
                        rows.extend([ig, ig])
                        cols.extend([ig, jg])
                        data.extend([value, -value])

                    else:  # boundary faces
                        bndry_id = -1 * (face.neighbor_id + 1)
                        bc = self.boundaries[bndry_id * self.n_groups + g]

                        # Geometric information
                        d_pf = (cell.centroid - face.centroid).norm()

                        # Boundary conditions
                        value = 0.0
                        if issubclass(type(bc), DirichletBoundary):
                            value = xs.D[g] / d_pf * face.area
                        elif issubclass(type(bc), RobinBoundary):
                            bc: RobinBoundary = bc
                            tmp = bc.a * d_pf - bc.b * xs.D[g]
                            value = bc.a * xs.D[g] / tmp * face.area

                        rows.append(ig)
                        cols.append(ig)
                        data.append(value)
        return csr_matrix((data, (rows, cols)),
                          shape=(fv.n_dofs(uk_man),) * 2)

    def scattering_matrix(self) -> csr_matrix:
        """Assemble the multigroup scattering matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

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

                    # Scattering term
                    value = xs.transfer_matrix[gp][g] * volume
                    rows.append(ig)
                    cols.append(igp)
                    data.append(value)
        return csr_matrix((data, (rows, cols)),
                          shape=(fv.n_dofs(uk_man),) * 2)

    def prompt_fission_matrix(self) -> csr_matrix:
        """Assemble the prompt multigroup fission matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

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
                        value = xs.chi[g] * xs.nu_sigma_f[gp] * volume
                    else:
                        # Prompt fission
                        value = xs.chi_prompt[g] * \
                                xs.nu_prompt_sigma_f[gp] * \
                                volume

                    rows.append(ig)
                    cols.append(igp)
                    data.append(value)
        return csr_matrix((data, (rows, cols)),
                          shape=(fv.n_dofs(uk_man),) * 2)

    def delayed_fission_matrix(self) -> csr_matrix:
        """Assemble the multigroup fission matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Construct if using precursors
        rows, cols, data = [], [], []
        if self.use_precursors:
            # Loop over cells
            for cell in self.mesh.cells:
                volume = cell.volume
                xs = self.material_xs[cell.material_id]

                # Loop over groups
                for g in range(self.n_groups):
                    ig = fv.map_dof(cell, 0, uk_man, 0, g)
                    for gp in range(self.n_groups):
                        igp = fv.map_dof(cell, 0, uk_man, 0, gp)

                        # Loop over precursors
                        value = 0.0
                        for j in range(xs.n_precursors):
                            value += xs.chi_delayed[g][j] * \
                                     xs.precursor_yield[j] * \
                                     xs.nu_delayed_sigma_f[gp] * \
                                     volume

                        rows.append(ig)
                        cols.append(igp)
                        data.append(value)
        return csr_matrix((data, (rows, cols)),
                          shape=(fv.n_dofs(uk_man),) * 2)

    def set_source(self) -> ndarray:
        """Assemble the right-hand side.

        Returns
        -------
        ndarray (n_cells * n_groups)
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Loop over cells
        b = np.zeros(fv.n_dofs(uk_man))
        for cell in self.mesh.cells:
            volume = cell.volume
            src = self.material_src[cell.material_id]

            # Loop over groups
            for g in range(self.n_groups):
                ig = fv.map_dof(cell, 0, uk_man, 0, g)
                b[ig] += src.values[g] * volume
        return b

    def apply_vector_bcs(self, b: ndarray) -> None:
        """Apply the boundary conditions to the right-hand side.

        Parameters
        ----------
        b : ndarray (n_cells * n_groups)
            The vector to apply boundary conditions to.
        """
        fv: FiniteVolume = self.discretization
        uk_man = self.phi_uk_man

        # Loop over boundary cells
        for bndry_id in self.mesh.boundary_cell_ids:
            cell = self.mesh.cells[bndry_id]
            xs = self.material_xs[cell.material_id]

            # Loop over faces
            for face in cell.faces:
                if not face.has_neighbor:
                    bndry_id = -1 * (face.neighbor_id + 1)

                    # Geometric information
                    d_pf = (cell.centroid - face.centroid).norm()

                    # Loop over groups
                    for g in range(self.n_groups):
                        bc = self.boundaries[bndry_id * self.n_groups + g]
                        ig = fv.map_dof(cell, 0, uk_man, 0, g)

                        # Boundary conditions
                        value = 0.0
                        if issubclass(type(bc), DirichletBoundary):
                            bc: DirichletBoundary = bc
                            value = xs.D[g] / d_pf * bc.value
                        elif issubclass(type(bc), NeumannBoundary):
                            bc: NeumannBoundary = bc
                            value = bc.value
                        elif issubclass(type(bc), RobinBoundary):
                            bc: RobinBoundary = bc
                            tmp = bc.a * d_pf - bc.b * xs.D[g]
                            value = -bc.b * xs.D[g] / tmp * bc.f

                        b[ig] += value * face.area

    def compute_precursors(self) -> None:
        """Compute the delayed neutron precursor concentrations.
        """
        if self.use_precursors:
            fv: FiniteVolume = self.discretization
            phi_uk_man = self.phi_uk_man
            c_uk_man = self.precursor_uk_man

            # Loop over cells
            self.precursors *= 0.0
            for cell in self.mesh.cells:
                xs = self.material_xs[cell.material_id]

                # Loop over precursors
                for j in range(xs.n_precursors):
                    ij = fv.map_dof(cell, 0, c_uk_man, 0, j)
                    coeff = xs.precursor_yield[j] / xs.precursor_lambda[j]

                    # Loop over groups
                    for g in range(self.n_groups):
                        ig = fv.map_dof(cell, 0, phi_uk_man, 0, g)
                        self.precursors[ij] += \
                            coeff * xs.nu_delayed_sigma_f[g] * self.phi[ig]

    def plot_solution(self, title: str = None) -> None:
        """Plot the solution, including the precursors, if used.

        Parameters
        ----------
        title : str
            A title for the figure.
        """
        fig: Figure = plt.figure()
        if self.use_precursors:
            if title:
                fig.suptitle(title)

            ax: Axes = fig.add_subplot(1, 2, 1)
            self.plot_flux(ax)

            ax: Axes = fig.add_subplot(1, 2, 2)
            self.plot_precursors(ax)
        else:
            ax: Axes = fig.add_subplot(1, 1, 1)
            self.plot_flux(ax, title)
        plt.tight_layout()

    def plot_flux(self, ax: Axes = None, title: str = None) -> None:
        """Plot the scalar flux on an Axes.

        Parameters
        ----------
        ax : Axes
            An Axes to plot on.
        title : str, default None
            A title for the Axes.
        """
        ax: Axes = plt.gca() if ax is None else ax
        if title:
            ax.set_title(title)

        grid = self.discretization.grid

        if self.mesh.dim == 1:
            grid = [p.z for p in grid]
            ax.set_xlabel("Location")
            ax.set_ylabel(r"$\phi(r)$")
            for g in range(self.n_groups):
                label = f"Group {g}"
                phi = self.phi[g::self.n_groups]
                ax.plot(grid, phi, label=label)
            ax.legend()
            ax.grid(True)
        elif self.mesh.dim == 2:
            x = np.unique([p.x for p in grid])
            y = np.unique([p.y for p in grid])
            xx, yy = np.meshgrid(x, y)
            phi: ndarray = self.phi[0::self.n_groups]
            phi = phi.reshape(xx.shape)
            im = ax.pcolor(xx, yy, phi, cmap="jet", shading="auto" ,
                           vmin=0.0, vmax=phi.max())
            plt.colorbar(im)
        plt.tight_layout()

    def plot_precursors(self, ax: Axes = None, title: str = None) -> None:
        """Plot the delayed neutron precursors on an Axes.

        Parameters
        ----------
        ax : Axes
            An Axes to plot on.
        title : str, default None
            A title for the Axes.
        """
        ax: Axes = plt.gca() if ax is None else ax
        if title:
            ax.set_title(title)

        if self.mesh.dim == 1:
            ax.set_xlabel("Location")
            ax.set_ylabel("Precursor Family")
            grid = [cell.centroid.z for cell in self.mesh.cells]
            for j in range(self.n_precursors):
                label = f"Family {j}"
                precursor = self.precursors[j::self.n_precursors]
                ax.plot(grid, precursor, label=label)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()

    def _check_inputs(self) -> None:
        self._check_mesh()
        self._check_discretization()
        self._check_boundaries()
        self._check_materials()

    def _check_mesh(self) -> None:
        if not self.mesh:
            raise AssertionError(
                "There must be a mesh attached to the solver.")
        if self.mesh.dim > 2:
            raise NotImplementedError(
                "Only 1D and 2D meshes are implemented.")

    def _check_discretization(self) -> None:
        if not self.discretization:
            raise AssertionError(
                "There must be a discretization attached to the solver.")
        if self.discretization.type not in ["FV", "PWC"]:
            raise NotImplementedError(
                "Only finite volume and piecewise continuous spatial "
                "discretizations are implemented.")

    def _check_boundaries(self) -> None:
        if not self.boundaries:
            raise AssertionError(
                "Boundary conditions must be attached to the solver.")
        if self.mesh.type == "LINE" and \
                len(self.boundaries) != 2 * self.n_groups:
            raise NotImplementedError(
                "There must be 2 * n_groups boundary conditions "
                "for 1D problems.")
        elif self.mesh.type == "ORTHO_QUAD" and \
                len(self.boundaries) != 4 * self.n_groups:
            raise NotImplementedError(
                "There must be 4 * n_groups boundary conditions "
                "for 2D problems.")

    def _check_materials(self) -> None:
        if not self.material_xs:
            raise AssertionError(
                "Material cross sections must be attached to the solver.")
        else:
            for xs in self.material_xs:
                if xs.n_groups != self.n_groups:
                    raise AssertionError(
                        "n_groups must agree across all cross sections.")

        if self.material_src:
            for n in range(len(self.material_xs)):
                if len(self.material_src) <= n + 1:
                    src = self.material_src[n]
                    if src.n_groups != self.n_groups:
                        raise AssertionError(
                            "All sources must be compatible with n_groups.")
                else:
                    src = MultiGroupSource(np.zeros(self.n_groups))
                    self.material_src.append(src)
        else:
            src = MultiGroupSource(np.zeros(self.n_groups))
            self.material_src = [src] * len(self.material_xs)