import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature

from ._harmonics import HarmonicIndex


class SteadyStateSolver:
    """
    Steady state multigroup neutron transport solver
    """

    from ._input_checks import (_check_mesh,
                                _check_discretization,
                                _check_materials,
                                _check_boundaries)

    from ._harmonics import create_harmonic_indices

    def __init__(self) -> None:
        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.n_moments: int = 0
        self.n_angles: int = 0
        self.max_precursors: int = 0

        self.scattering_order: int = 0
        self.use_precursors: bool = False

        # Domain objects
        self.mesh: Mesh = None
        self.discretization: SpatialDiscretization = None
        self.boundaries: List[Boundary] = []

        # Materials objects
        self.materials: List[Material] = []

        self.material_xs: List[CrossSections] = []
        self.material_src: List[IsotropicMultiGroupSource] = []
        self.matid_to_xs_map: List[int] = []
        self.matid_to_src_map: List[int] = []
        self.cellwise_xs: List[LightWeightCrossSections] = []

        # Quadrature information
        self.quadrature: ProductQuadrature = None
        self.harmonic_index_map: List[HarmonicIndex] = []

        # Flux moment information
        self.phi: ndarray = None
        self.phi_prev: ndarray = None
        self.phi_uk_man: UnknownManager = None

        # Angular flux information
        self.psi: ndarray = None
        self.psi_uk_man: UnknownManager = None

        # Precursor information
        self.precursors: ndarray = None

        # Angular operators
        self.M: ndarray = None
        self.D: ndarray = None

    def initialize(self) -> None:
        """
        Initialize the solver.
        """
        self._check_inputs()
        sd = self.discretization

        # Define the harmonic index map
        self.create_harmonic_indices()

        # Define number of moments and angles
        self.n_angles = len(self.quadrature.abscissae)
        self.n_moments = len(self.harmonic_index_map)

        # Initialize flux moments
        self.phi_uk_man = UnknownManager()
        for m in range(self.n_moments):
            self.phi_uk_man.add_unknown(self.n_groups)
        self.phi = np.zeros(sd.n_dofs(self.phi_uk_man))
        self.phi_prev = np.zeros(sd.n_dofs(self.phi_uk_man))

        # Initialize angular flux
        self.psi_uk_man = UnknownManager()
        for n in range(self.n_angles):
            self.psi_uk_man.add_unknown(self.n_groups)
        self.psi = np.zeros(sd.n_dofs(self.psi_uk_man))

        # Initialize precursors
        if self.use_precursors:
            n_dofs = self.n_precursors * self.mesh.n_cells
            self.precursors = np.zeros(n_dofs)

        # Initialize harmonics
        self.create_harmonic_indices()

    def _check_inputs(self) -> None:
        """
        Check the inputs.
        """
        self._check_mesh()
        self._check_discretization()
        self._check_materials()
        self._check_boundaries()
