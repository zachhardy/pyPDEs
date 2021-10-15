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
        self.material_src: List[MultiGroupSource] = []
        self.cellwise_xs: List[LightWeightCrossSections] = []

        # Unknown managers
        self.phi_uk_man: UnknownManager = None
        self.psi_uk_man: UnknownManager = None

        # Unknown storage vectors
        self.phi: ndarray = None
        self.phi_ell: ndarray = None
        self.psi: ndarray = None
        self.precursors: ndarray = None

    def initialize(self) -> None:
        """Initialize the neutron transport solver.
        """
        pass

    def _check_inputs(self) -> None:
        pass