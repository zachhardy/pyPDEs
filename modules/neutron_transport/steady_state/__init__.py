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
from pyPDEs.utilities.quadratures import AngularQuadrature
from pyPDEs.utilities.boundaries import Boundary


class SteadyStateSolver:
    """Steady state neutron transport solver.
    """

    from ._input_checks import (_check_mesh,
                                _check_discretization,
                                _check_materials,
                                _check_boundaries)

    def __init__(self) -> None:
        """Constructor.
        """
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

        # Quadrature
        self.quadrature: AngularQuadrature = None

        # Materials information
        self.material_xs: List[CrossSections] = []
        self.material_src: List[IsotropicMultiGroupSource] = []
        self.cellwise_xs: List[LightWeightCrossSections] = []

        # Unknown managers
        self.phi_uk_man: UnknownManager = None
        self.psi_uk_man: UnknownManager = None

        # Unknown storage vectors
        self.phi: ndarray = None
        self.phi_ell: ndarray = None
        self.psi: ndarray = None
        self.precursors: ndarray = None

        # Iteration parameters
        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 500

    def initialize(self) -> None:
        """Initialize the neutron transport solver.
        """
        self._check_inputs()

    def _check_inputs(self) -> None:
        self._check_mesh()
        self._check_discretization()
        self._check_materials()
        self._check_boundaries()