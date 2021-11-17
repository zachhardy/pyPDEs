import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature


class SteadyStateSolver:
    """
    Steady state multigroup neutron transport solver
    """

    from ._input_checks import (_check_mesh,
                                _check_discretization,
                                _check_materials,
                                _check_boundaries)

    def __init__(self) -> None:
        self.n_groups: int = 0
        self.n_moments: int = 0
        self.n_precursors: int = 0
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

        # Quadrature object
        self.quadrature: ProductQuadrature = None

        # Flux moment information
        self.phi: ndarray = None
        self.phi_prev: ndarray = None
        self.phi_uk_man: UnknownManager = None

        # Angular flux information
        self.psi: ndarray = None
        sel.psi_uk_man: UnknownManager = None

        # Precursor information
        self.precursors: ndarray = None

    def initialize(self) -> None:
        """
        Initialize the solver.
        """
        self._check_inputs()

    def _check_inputs(self) -> None:
        """
        Check the inputs.
        """
        self._check_mesh()
        self._check_discretization()
        self._check_materials()
        self._check_boundaries()
