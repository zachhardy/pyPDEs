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
    """
    Class for solving multigroup diffusion problems.
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

        self.b: ndarray = None
        self.L: List[csr_matrix] = None

        self.phi: ndarray = None
        self.flux_uk_man: UnknownManager = UnknownManager()

        self.use_precursors: bool = False

        self.precursors: ndarray = None
        self.precursor_uk_man: UnknownManager = UnknownManager()

        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

    @property
    def n_groups(self) -> int:
        """
        Get the number of groups.
        """
        return self.material_xs[0].n_groups

    @property
    def n_precursors(self) -> int:
        """
        Get the total number of precursors.

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

        # ======================================== Initialize system storage
        self.b = np.zeros(flux_dofs)
        self.L = []
        for g in range(self.n_groups):
            if isinstance(self.discretization, FiniteVolume):
                self.L.append(self.fv_assemble_matrix(g))
            else:
                self.L.append(self.pwc_assemble_matrix(g))

    def execute(self) -> None:
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
                if isinstance(self.discretization, FiniteVolume):
                    self.fv_set_source(g, self.phi)
                else:
                    self.pwc_set_source(g, self.phi)
                self.phi[g::n_grps] = spsolve(self.L[g],
                                              self.b[g::n_grps])

            # =================================== Check convergence
            phi_change = norm(self.phi - phi_ell)
            phi_ell[:] = self.phi
            if phi_change <= self.tolerance:
                converged = True
                break

        # ======================================== Compute precursors
        if self.use_precursors:
            if isinstance(self.discretization, FiniteVolume):
                self.fv_compute_precursors()
            else:
                self.pwc_compute_precursors()

        # ======================================== Print summary
        if converged:
            msg = '***** Solver Converged *****'
        else:
            msg = '!!!!! WARNING: Solver NOT Converged !!!!!'
        msg += f'\nFinal Change:\t\t{phi_change:.3e}'
        msg += f'\n# of Iterations:\t{nit}'
        print(msg)
        print("\n***** Done executing steady-state "
              "multi-group diffusion solver. *****\n")

    def plot_solution(self, title: str = None) -> None:
        """
        Plot the solution, including the precursors, if used.

        Parameters
        ----------
        title : str
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
        plt.tight_layout()

    def plot_precursors(self, ax: Axes = None, title: str = None) -> None:
        """
        Plot the delayed neutron precursors on ax.

        Parameters
        ----------
        ax : Axes
            An Axes to plot on.
        title : str, default None
        """
        ax: Axes = plt.gca() if ax is None else ax
        if title:
            ax.set_title(title)

        grid = self.discretization.grid

        if self.mesh.dim == 1:
            ax.set_xlabel("Location")
            ax.set_ylabel("Precursor Family")
            grid = [p.z for p in grid]
            for j in range(self.n_precursors):
                label = f"Family {j}"
                precursor = self.precursors[j::self.n_precursors]
                ax.plot(grid, precursor, label=label)
            ax.legend()
            ax.grid(True)
        plt.tight_layout()

    def check_inputs(self) -> None:
        self._check_mesh()
        self._check_discretization()
        self._check_boundaries()
        self._check_materials()

    def _check_mesh(self) -> None:
        if not self.mesh:
            raise AssertionError("No mesh is attached to the solver.")
        elif self.mesh.dim != 1:
            raise NotImplementedError(
                "Only 1D problems have been implemented.")

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
        elif len(self.boundaries) != 2 * self.n_groups:
            raise NotImplementedError(
                "There can only be 2 * n_groups boundary conditions "
                "for 1D problems.")

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
                if len(self.material_src) < n + 1:
                    src = self.material_src[n]
                    if src.n_components != self.n_groups:
                        raise AssertionError(
                            "All sources must be compatible with num_groups.")
                else:
                    src = MultiGroupSource(np.zeros(self.n_groups))
                    self.material_src.append(src)
        else:
            src = MultiGroupSource(np.zeros(self.n_groups))
            self.material_src = [src] * len(self.material_xs)
