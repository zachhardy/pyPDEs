import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from numpy.linalg import norm
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from typing import List

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import Boundary


class SteadyStateSolver:
    """Steady-state multigroup diffusion
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

    from ._input_checks import (_check_mesh,
                                _check_discretization,
                                _check_boundaries,
                                _check_materials)

    def __init__(self) -> None:
        """Class constructor.
        """
        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.max_precursors: int = 0

        self.use_precursors: bool = False

        # Domain objects
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = []

        # Materials information
        self.materials: List[Material] = []

        self.material_xs: List[CrossSections] = []
        self.material_src: List[MultiGroupIsotropicSource] = []
        self.matid_to_xs_map: List[int] = []
        self.matid_to_src_map: List[int] = []
        self.cellwise_xs: List[LightWeightCrossSections] = []

        # Precomputed matrices
        self.L: csr_matrix = None
        self.S: csr_matrix = None
        self.Fp: csr_matrix = None
        self.Fd: csr_matrix = None

        # Scalar flux solution vector
        self.phi: ndarray = None
        self.phi_uk_man: UnknownManager = None

        # Precursor solution vector
        self.precursors: ndarray = None

    def initialize(self) -> None:
        """Initialize the solver.
        """
        self._check_inputs()
        sd = self.discretization

        # Initialize phi information
        self.phi_uk_man = UnknownManager()
        self.phi_uk_man.add_unknown(self.n_groups)
        self.phi = np.zeros(sd.n_dofs(self.phi_uk_man))

        if self.use_precursors:
            n_dofs = self.n_precursors * self.mesh.n_cells
            self.precursors = np.zeros(n_dofs)

        # Initialize cell-wise cross sections
        self.cellwise_xs.clear()
        for cell in self.mesh.cells:
            xs = self.material_xs[cell.material_id]
            self.cellwise_xs += [LightWeightCrossSections(xs)]

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

    def diffusion_matrix(self) -> csr_matrix:
        """Assemble the multigroup diffusion matrix.

        This routine assembles the diffusion plus interaction matrix
        for all groups according to the DoF ordering of `phi_uk_man`.

        Returns
        -------
        csr_matrix (n_cells * n_groups,) * 2
        """
        if isinstance(self.discretization, FiniteVolume):
            return self._fv_diffusion_matrix()
        elif isinstance(self.discretization, PiecewiseContinuous):
            return self._pwc_diffusion_matrix()

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
        self._check_materials()
        self._check_boundaries()
