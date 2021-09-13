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
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import *

class SteadyStateSolver:
    """Class for solving steady-state multigroup diffusion problems.
    """

    from ._fv import (_fv_diffusion_matrix,
                      _fv_scattering_matrix,
                      _fv_prompt_fission_matrix,
                      _fv_delayed_fission_matrix,
                      _fv_set_source,
                      _fv_apply_matrix_bcs,
                      _fv_apply_vector_bcs,
                      _fv_compute_precursors)

    from ._pwc import (_pwc_diffusion_matrix,
                       _pwc_scattering_matrix,
                       _pwc_prompt_fission_matrix,
                       _pwc_delayed_fission_matrix,
                       _pwc_set_source,
                       _pwc_apply_matrix_bcs,
                       _pwc_apply_vector_bcs,
                       _pwc_compute_precursors)

    from ._plotting import (plot_solution,
                            plot_flux,
                            plot_precursors)

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
                n = max_precursors * self.mesh.n_cells
                self.precursor_uk_man.add_unknown(max_precursors)
                self.precursors = np.zeros(n)
            else:
                self.use_precursors = False

        # Precompute matrices
        self.L = self.diffusion_matrix()
        self.S = self.scattering_matrix()
        self.Fp = self.prompt_fission_matrix()
        if self.use_precursors:
            self.Fd = self.delayed_fission_matrix()

    def execute(self) -> None:
        """Execute the steady-state multigroup diffusion solver.
        """
        A = self.assemble_matrix()
        b = self.assemble_rhs()
        self.phi = spsolve(A, b)
        if self.use_precursors:
            self.compute_precursors()

    def assemble_matrix(self) -> csr_matrix:
        A = self.L - self.S - self.Fp
        if self.use_precursors:
            A -= self.Fd
        return self.apply_matrix_bcs(A)

    def assemble_rhs(self) -> ndarray:
        b = self.set_source()
        return self.apply_vector_bcs(b)

    def diffusion_matrix(self, t: float = 0.0) -> csr_matrix:
        """Assemble the multigroup diffusion matrix.

        This routine assembles the diffusion plus interaction matrix
        for all groups according to the DoF ordering of `phi_uk_man`.

        Parameters
        ----------
        t : float, default 0.0
            The simulation time.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_diffusion_matrix(t)
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_diffusion_matrix(t)

    def scattering_matrix(self) -> csr_matrix:
        """Assemble the multigroup scattering matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_scattering_matrix()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_scattering_matrix()

    def prompt_fission_matrix(self) -> csr_matrix:
        """Assemble the prompt multigroup fission matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_prompt_fission_matrix()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_prompt_fission_matrix()

    def delayed_fission_matrix(self) -> csr_matrix:
        """Assemble the multigroup fission matrix.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_delayed_fission_matrix()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_delayed_fission_matrix()

    def set_source(self) -> ndarray:
        """Assemble the right-hand side.

        Returns
        -------
        ndarray (n_cells * n_groups)
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_set_source()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_set_source()

    def apply_matrix_bcs(self, A: csr_matrix) -> csr_matrix:
        """Apply the boundary conditions to a matrix.

        Parameters
        ----------
        A : csr_matrix (n_cells * n_groups,) * 2

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
            The input matrix with boundary conditions applied.
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_apply_matrix_bcs(A)
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_apply_matrix_bcs(A)

    def apply_vector_bcs(self, b: ndarray) -> ndarray:
        """Apply the boundary conditions to the right-hand side.

        Parameters
        ----------
        b : ndarray (n_cells * n_groups)
            The vector to apply boundary conditions to.

        Returns
        -------
        ndarray (n_cells * n_groups)
            The input vector with boundary conditions applied.
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_apply_vector_bcs(b)
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_apply_vector_bcs(b)

    def compute_precursors(self) -> None:
        """Compute the delayed neutron precursor concentrations.
        """
        if isinstance(self.discretization, FiniteVolume):
            self._fv_compute_precursors()
        elif isinstance(self.discretization, PiecewiseContinuous):
            self._pwc_compute_precursors()

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
                len(self.boundaries) != 2:
            raise NotImplementedError(
                "There must be 2 boundary conditions for 1D problems.")
        elif self.mesh.type == "ORTHO_QUAD" and \
                len(self.boundaries) != 4:
            raise NotImplementedError(
                "There must be 4 boundary conditions for 2D problems.")
        for b, bc in enumerate(self.boundaries):
            error = False
            if issubclass(type(bc), DirichletBoundary):
                bc: DirichletBoundary = bc
                if len(bc.values) != self.n_groups:
                    error = True
            elif issubclass(type(bc), NeumannBoundary):
                bc: NeumannBoundary = bc
                if len(bc.values) != self.n_groups:
                    error = True
            elif issubclass(type(bc), RobinBoundary):
                bc: RobinBoundary = bc
                vals = [bc.a, bc.b, bc.f]
                if any([len(v) != self.n_groups for v in vals]):
                    error = True
            if error:
                raise AssertionError(
                    f"Invalid number of components found in boundary {b}.")

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