import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pyPDEs.mesh import Mesh
from pyPDEs.math.discretization import FiniteVolume

from pyPDEs.material import Material
from pyPDEs.material import CrossSections
from pyPDEs.material import IsotropicMultiGroupSource
from pyPDEs.material import LightWeightCrossSections

from typing import Union, Callable
from pyPDEs.mesh import CartesianVector

BCFunc = Callable[[CartesianVector, float], float]
BCType = Union[float, BCFunc]

from ..boundaries import Boundary


class SteadyStateSolver:
    """
    Implementation of a steady-state multi-group diffusion solver.
    """

    from ._initialize import _initialize_materials
    from ._initialize import _initialize_boundaries

    from ._assemble_matrix import _assemble_matrix
    from ._assemble_rhs import _assemble_rhs

    from ._write import write_scalar_flux
    from ._write import write_precursors
    from ._write import write_fission_rate

    def __init__(
            self,
            discretization: FiniteVolume,
            materials: list[Material],
            boundary_info: list[tuple[str, int]],
            boundary_values: list[list[list[BCType]]] = None
    ) -> None:
        """
        Construct the solver from a discretization, list of materials,
        and boundary condition inputs.

        Parameters
        ----------
        discretization : FiniteVolume
        materials : list[Material]
        boundary_info : list[tuple[str, int]]
            A list containing the boundary condition types and the
            corresponding index of the boundary values list where
            the boundary values for the particular boundary condition
            can be found.
        boundary_values : list[list[list[float]]], default None+
            The boundary values. The outer list represents values for
            a particular boundary, the middle the values for a particular
            energy group, and the inner the actual boundary values. This
            is a list and not a float to allow for Robin boundaries which
            use three values. The default value for this is None,
            corresponding to no boundary values.

        Notes
        -----
        The mesh for the problem is obtained from the specified
        discretization and all other features such as boundary
        """

        # ==================== Spatial Domain ==================== #

        self.mesh: Mesh = discretization.mesh
        self.discretization: FiniteVolume = discretization

        # ==================== Materials ====================#

        self.materials: list[Material] = materials
        self.material_xs: list[CrossSections] = []
        self.material_src: list[IsotropicMultiGroupSource] = []
        self.cellwise_xs: list[LightWeightCrossSections] = []

        self.matid_to_xs_map: list[int] = []
        self.matid_to_src_map: list[int] = []

        # ==================== Boundary Conditions ====================#

        self.boundary_info: list[tuple[str, int]] = boundary_info
        self.boundary_values: list[list[list[BCType]]] = boundary_values

        self.boundaries: list[list[Boundary]] = []

        # ==================== Problem Information ====================#

        self.n_groups: int = 0

        self.n_precursors: int = 0
        self.max_precursors: int = 0
        self.use_precursors: bool = False

        self.phi: np.ndarray = None
        self.phi_ell: np.ndarray = None

        self.precursors: np.ndarray = None

        self._A: list[csr_matrix] = None
        self._b: np.ndarray = None

    def initialize(self) -> None:
        """
        Initialize the solver.
        """

        msg = "Initializing the multi-group diffusion solver"
        msg = "\n".join(["", "*" * len(msg), msg, "*" * len(msg), ""])
        print(msg)

        # ==================================================
        # Check the mesh
        # ==================================================

        if self.mesh.dimension > 2:
            msg = "Only 2D problems have been implemented."
            raise AssertionError(msg)

        # ==================================================
        # Check the discretization
        # ==================================================

        if self.discretization.type not in ["FV"]:
            msg = "Only finite volume discretizations " \
                  "have been implemented."
            raise AssertionError(msg)

        # ==================================================
        # Initialize the material properties
        # ==================================================

        self._initialize_materials()

        # ==================================================
        # Initialize the boundary conditions
        # ==================================================

        self._initialize_boundaries()

        # ==================================================
        # Initialize data storage
        # ==================================================

        n_phi_dofs = len(self.mesh.cells) * self.n_groups
        self.phi = np.zeros(n_phi_dofs)
        self.phi_ell = np.zeros(n_phi_dofs)

        if self.use_precursors:
            n_precursor_dofs = len(self.mesh.cells) * self.max_precursors
            self.precursors = np.zeros(n_precursor_dofs)

        self._b = np.zeros(self.phi.shape)

    def execute(self) -> None:
        """
        Execute the multi-group diffusion solver.
        """

        msg = "Executing the steady-state multi-group diffusion solver"
        msg = "\n".join(["", "*" * len(msg), msg, "*" * len(msg), ""])
        print(msg)

        self._assemble_matrix(with_scattering=True,
                              with_fission=True)
        self._assemble_rhs(with_material_src=True,
                           with_boundary_src=True)
        self.phi = spsolve(self._A[0], self._b)
        if self.use_precursors:
            self._compute_precursors()

    def _compute_precursors(self) -> None:
        """
        Compute the steady-state delayed neutron precursor concentrations.
        """
        if not self.use_precursors:
            return

        self.precursors[:] = 0.0
        for cell in self.mesh.cells:
            xs_id = self.matid_to_xs_map[cell.material_id]
            xs = self.material_xs[xs_id]

            if xs.is_fissile:
                uk_map_phi = self.n_groups * cell.id
                uk_map_precursor = self.max_precursors * cell.id

                for j in range(xs.n_precursors):
                    value = 0.0
                    for g in range(self.n_groups):
                        value += (xs.precursor_yield[j] /
                                  xs.precursor_lambda[j] *
                                  xs.nu_delayed_sigma_f[g] *
                                  self.phi[uk_map_phi + g])
                    self.precursors[uk_map_precursor + j] = value
