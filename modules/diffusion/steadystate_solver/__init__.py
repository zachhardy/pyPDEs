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

    Attributes
    ----------
    mesh : Mesh
        The spatial mesh to solve the problem on.
    discretization : SpatialDiscretization
        The spatial discretization used to solve the problem.
    boundaries : List[Boundary]
        The boundary conditions imposed on the equations.
        There should be a boundary condition for each group
        and boundary. In the list, each boundaries group-wise
        boundary conditions should be listed next to each other.
    material_xs : List[CrossSections]
        The cross sections corresponding to the material IDs
        defined on the cells. There should be as many cross
        sections as unique material IDs on the mesh.
    material_src : List[MultigroupSource]
        The multi-group sources corresponding to the material
        IDs defined on the cells. There should be as many
        multi-group sources as unique material IDs on the mesh.
    use_precursors : bool
        A flag for including delayed neutrons.
    tolerance : float
        The iterative tolerance for the group-wise solver.
    max_iterations : int
        The maximum number of iterations for the group-wise
        solver to take before exiting.
    b : ndarray (n_nodes * n_groups,)
        The right-hand side of the linear system to solve.
    L : List[csr_matrix]
        The group-wise diffusion operators used to solve the
        equations group-wise. There are n_groups matrices stored.
    phi : ndarray (n_nodes * n_groups,)
        The most current scalar flux solution vector.
    flux_uk_man : UnknownManager
        An unknown manager tied to the scalar flux solution vector.
    precurosrs : ndarray (n_nodes * max_precursors_per_material,)
        The delayed neutron precursor concentrations.

        In multi-material problems, this vector stores up to the
        maximum number of precursors that live on any given material.
        This implies that material IDs must be used to map the
        concentration of specific precursor species. This structure
        is used to prevent very sparse vectors in many materials.
    precursor_uk_man : UnknownManager
        An unknown manager tied to the precursor vector.
    """

    from .assemble_fv import fv_assemble_matrix
    from .assemble_fv import fv_set_source
    from .assemble_fv import fv_compute_precursors

    from .assemble_pwc import pwc_assemble_matrix
    from .assemble_pwc import pwc_set_source
    from .assemble_pwc import pwc_compute_precursors

    def __init__(self) -> None:
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = None

        self.material_xs: List[CrossSections] = None
        self.material_src: List[MultiGroupSource] = None

        self.use_precursors: bool = False

        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

        self.b: ndarray = None
        self.L: List[csr_matrix] = None

        self.phi: ndarray = None
        self.flux_uk_man: UnknownManager = UnknownManager()

        self.precursors: ndarray = None
        self.precursor_uk_man: UnknownManager = UnknownManager()

    @property
    def n_groups(self) -> int:
        """
        Get the number of groups.

        Returns
        -------
        int
        """
        return self.material_xs[0].n_groups

    @property
    def n_precursors(self) -> int:
        """
        Get the total number of precursors.

        Returns
        -------
        int
        """
        return sum([xs.n_precursors for xs in self.material_xs])

    def initialize(self) -> None:
        """
        Initialize the diffusion solver.
        """
        self.check_inputs()
        sd = self.discretization

        # ======================================== Initialize flux
        self.flux_uk_man.clear()
        self.flux_uk_man.add_unknown(self.n_groups)
        flux_dofs = sd.n_dofs(self.flux_uk_man)
        self.phi = np.zeros(flux_dofs)

        # ======================================== Initialize precursors
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

        # ======================================== Initialize system storage
        self.b = np.zeros(flux_dofs)
        self.L = []
        for g in range(self.n_groups):
            self.L.append(self.assemble_matrix(g))

    def execute(self, verbose: bool = False) -> None:
        """
        Execute the steady-state diffusion solver.
        """
        print("\n***** Executing steady-state "
              "multi-group diffusion solver *****\n")
        n_grps = self.n_groups
        phi_ell = np.zeros(self.phi.shape)
        phi_change = 1.0

        # ======================================== Start iterating
        converged = False
        for nit in range(self.max_iterations):

            # =================================== Solve group-wise
            self.b *= 0.0
            for g in range(n_grps):
                self.set_source(g, self.phi)
                self.phi[g::n_grps] = \
                    spsolve(self.L[g], self.b[g::n_grps])

            # =================================== Check convergence
            phi_change = norm(self.phi - phi_ell)
            phi_ell[:] = self.phi

            if verbose:
                print(f"===== Iteration {nit} Change = {phi_change:.3e}")

            if phi_change <= self.tolerance:
                converged = True
                break

        # ======================================== Compute precursors
        self.compute_precursors()

        # ======================================== Print summary
        if converged:
            msg = "\n***** Solver Converged *****"
        else:
            msg = "!!!!! WARNING: Solver NOT Converged !!!!!"
        msg += f"\nFinal Change:\t\t{phi_change:.3e}"
        msg += f"\n# of Iterations:\t{nit}"
        print(msg)
        print("\n***** Done executing steady-state "
              "multi-group diffusion solver. *****\n")

    def assemble_matrix(self, g: int) -> csr_matrix:
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
        if isinstance(self.discretization, FiniteVolume):
            return self.fv_assemble_matrix(g)
        else:
            return self.pwc_assemble_matrix(g)

    def set_source(self, g: int, phi: ndarray,
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

        flags = (apply_material_source, apply_scattering,
                 apply_fission, apply_boundaries)
        if isinstance(self.discretization, FiniteVolume):
            self.fv_set_source(g, phi, *flags)
        else:
            self.pwc_set_source(g, phi, *flags)

    def compute_precursors(self) -> None:
        """
        Compute the delayed neutron precursor concentration.
        """
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                self.fv_compute_precursors()
            else:
                self.pwc_compute_precursors()

    def plot_solution(self, title: str = None) -> None:
        """
        Plot the solution, including the precursors, if used.

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
        """
        Plot the scalar flux on ax.

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
        """
        Plot the delayed neutron precursors on ax.

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

    def check_inputs(self) -> None:
        """
        Check the inputs provided to the solver.
        """
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
