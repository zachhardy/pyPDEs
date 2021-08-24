import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from numpy import ndarray
from scipy.sparse import csr_matrix
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List

from pyPDEs.mesh import Mesh
from pyPDEs.material import CrossSections, MultiGroupSource
from pyPDEs.spatial_discretization import (SpatialDiscretization,
                                           FiniteVolume)
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import Boundary


class SteadyStateSolver:
    """Class for solving multigroup diffusion problems.
    """

    from ._assemble_fv import _fv_assemble_diffusion_matrix
    from ._assemble_fv import _fv_assemble_scattering_matrix
    from ._assemble_fv import _fv_assemble_fission_matrix
    from ._assemble_fv import _fv_set_source
    from ._assemble_fv import _fv_compute_precursors

    from ._assemble_pwc import _pwc_assemble_diffusion_matrix
    from ._assemble_pwc import _pwc_assemble_scattering_matrix
    from ._assemble_pwc import _pwc_assemble_fission_matrix
    from ._assemble_pwc import _pwc_set_source
    from ._assemble_pwc import _pwc_compute_precursors

    def __init__(self) -> None:
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = None

        self.material_xs: List[CrossSections] = None
        self.material_src: List[MultiGroupSource] = None

        self.use_precursors: bool = False

        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

        self.use_groupwise_solver: bool = True

        self.L: csr_matrix = None
        self.S: csr_matrix = None
        self.F: csr_matrix = None
        # self.Lg: List[csr_matrix] = []

        self.b: ndarray = None

        self.phi: ndarray = None
        self.flux_uk_man: UnknownManager = UnknownManager()

        self.precursors: ndarray = None
        self.precursor_uk_man: UnknownManager = UnknownManager()

    @property
    def n_groups(self) -> int:
        """Get the number of groups.

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
        return sum([xs.n_precursors for xs in self.material_xs])

    def initialize(self) -> None:
        """Initialize the steady state diffusion solver.
        """
        self._check_inputs()
        sd = self.discretization

        # Initialize flux
        self.flux_uk_man.clear()
        self.flux_uk_man.add_unknown(self.n_groups)
        flux_dofs = sd.n_dofs(self.flux_uk_man)
        self.phi = np.zeros(flux_dofs)

        # Initialize precursors
        if self.use_precursors:
            # Determine the max precursors per material
            max_precursors = 0
            for xs in self.material_xs:
                if xs.n_precursors > max_precursors:
                    max_precursors = xs.n_precursors

            # Set unknown manager and vectors
            if self.n_precursors > 0:
                self.precursor_uk_man.clear()
                self.precursor_uk_man.add_unknown(max_precursors)
                precursor_dofs = self.mesh.n_cells * max_precursors
                self.precursors = np.zeros(precursor_dofs)
            else:
                self.use_precursors = False

        #  Initialize system storage
        self.b = np.zeros(flux_dofs)

        self.L = self.assemble_diffusion_matrix()
        self.S = self.assemble_scattering_matrix()
        self.F = self.assemble_fission_matrix()

    def execute(self, verbose: bool = False) -> None:
        """Execute the steady-state diffusion solver.
        """
        print("\n***** Executing steady-state "
              "multi-group diffusion solver *****")

        # Solve the full multi-group system
        if not self.use_groupwise_solver:
            self.b *= 0.0
            self.set_source(True, True, False, False)
            A = self.L - self.S - self.F
            self.phi = spsolve(A, self.b)

        # Solve the system group-wise
        else:
            n_grps = self.n_groups
            phi_ell = np.zeros(self.phi.shape)
            phi_change = 1.0

            # Start iterating
            converged = False
            for nit in range(self.max_iterations):

                # Solve group-wise
                for g in range(n_grps):
                    self.b *= 0.0
                    self.set_source()
                    self.phi[g::n_grps] = \
                        spsolve(self.Lg(g), self.bg(g))

                # Check convergence
                phi_change = norm(self.phi - phi_ell)
                phi_ell[:] = self.phi

                if verbose:
                    print(f"===== Iteration {nit} Change = {phi_change:.3e}")

                if phi_change <= self.tolerance:
                    converged = True
                    break

            # Compute precursors
            self.compute_precursors()

            # Print summary
            if converged:
                msg = "\n***** Solver Converged *****"
            else:
                msg = "!!!!! WARNING: Solver NOT Converged !!!!!"
            msg += f"\nFinal Change:\t\t{phi_change:.3e}"
            msg += f"\n# of Iterations:\t{nit}"
            print(msg)

    def assemble_diffusion_matrix(self) -> csr_matrix:
        """Assemble the multi-group diffusion matrix.

        Returns
        -------
        csr_matrix
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_assemble_diffusion_matrix()
        else:
            return self._pwc_assemble_diffusion_matrix()

    def assemble_scattering_matrix(self) -> csr_matrix:
        """Assemble the multi-group scattering matrix.

        Returns
        -------
        csr_matrix
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_assemble_scattering_matrix()
        else:
            return self._pwc_assemble_scattering_matrix()

    def assemble_fission_matrix(self) -> csr_matrix:
        """Assemble the multi-group fission matrix.

        Returns
        -------
        csr_matrix
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_assemble_fission_matrix()
        else:
            return self._pwc_assemble_fission_matrix()

    def Lg(self, g: int) -> csr_matrix:
        """Get the `g`'th group's diffusion matrix.

        Parameters
        ----------
        g : int

        Returns
        -------
        csr_matrix
        """
        if self.flux_uk_man.storage_method == "NODAL":
            return self.L[g::self.n_groups, g::self.n_groups]
        else:
            ni = g * self.discretization.n_nodes
            nf = (g + 1) * self.discretization.n_nodes
            return self.L[ni:nf, ni:nf]

    def bg(self, g: int) -> ndarray:
        """Get the `g`'th group's right-hand side.

        Parameters
        ----------
        g : int

        Returns
        -------
        ndarray
        """
        if self.flux_uk_man.storage_method == "NODAL":
            return self.b[g::self.n_groups]
        else:
            ni = g * self.discretization.n_nodes
            nf = (g + 1) * self.discretization.n_nodes
            return self.b[ni:nf]

    def set_source(self,
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
        flags = (apply_material_source, apply_boundary_source,
                 apply_scattering_source, apply_fission_source)
        if isinstance(self.discretization, FiniteVolume):
            self._fv_set_source(*flags)
        else:
            self._pwc_set_source(*flags)

    def compute_precursors(self) -> None:
        """Compute the delayed neutron precursor concentration.
        """
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                self._fv_compute_precursors()
            else:
                self._pwc_compute_precursors()

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
            raise AssertionError("No mesh is attached to the solver.")
        if self.mesh.dim > 2:
            raise NotImplementedError(
                "Only 1D and 2D problems have been implemented.")

    def _check_discretization(self) -> None:
        if not self.discretization:
            raise AssertionError(
                "No discretization is attached to the solver.")
        elif self.discretization.type not in ["FV", "PWC"]:
            raise NotImplementedError(
                "Only finite volume has been implemented.")

    def _check_boundaries(self) -> None:
        if not self.boundaries:
            raise AssertionError(
                "No boundary conditions are attached to the solver.")
        if self.mesh.type == "LINE" and \
                len(self.boundaries) != 2 * self.n_groups:
            raise NotImplementedError(
                "There can only be 2 * n_groups boundary conditions "
                "for 1D problems.")
        if self.mesh.type == "ORTHO_QUAD" and \
                len(self.boundaries) != 4 * self.n_groups:
            raise NotImplementedError(
                "There can only be 4 * n_groups boundary conditions "
                "for 2D problems.")

    def _check_materials(self) -> None:
        if not self.material_xs:
            raise AssertionError(
                "No material cross section are attached to the solver.")
        else:
            for xs in self.material_xs:
                if xs.n_groups != self.n_groups:
                    raise AssertionError(
                        "num_groups must agree across all cross sections.")

        if self.material_src:
            for n in range(len(self.material_xs)):
                if len(self.material_src) <= n + 1:
                    src = self.material_src[n]
                    if src.n_groups != self.n_groups:
                        raise AssertionError(
                            "All sources must be compatible with num_groups.")
                else:
                    src = MultiGroupSource(np.zeros(self.n_groups))
                    self.material_src.append(src)
        else:
            src = MultiGroupSource(np.zeros(self.n_groups))
            self.material_src = [src] * len(self.material_xs)
