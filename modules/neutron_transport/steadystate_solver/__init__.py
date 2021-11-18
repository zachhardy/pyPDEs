import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature

from ..boundaries import *
from ._harmonics import HarmonicIndex


class SteadyStateSolver:
    """
    Steady state multigroup neutron transport solver
    """

    from ._check_inputs import check_inputs
    from ._initialize import (initialize_materials,
                              initialize_unknowns,
                              initialize_bondaries,
                              compute_n_moments,
                              compute_n_angles)
    from ._harmonics import create_harmonic_indices
    from ._angular_operators import (discrete_to_moment_matrix,
                                     moment_to_discrete_matrix)
    from ._setsource import set_source

    def __init__(self) -> None:
        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.n_moments: int = 0
        self.n_angles: int = 0
        self.max_precursors: int = 0

        self.scattering_order: int = 0
        self.use_precursors: bool = False

        # Iteration parameters
        self.tolerance: float = 1.0e-8
        self.max_iterations: int = 100

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
        self.harmonic_index_map: List[HarmonicIndex] = []

        # Unknown managers
        self.phi_uk_man: UnknownManager = None
        self.psi_uk_man: UnknownManager = None

        # Flux moment vectors
        self.phi: ndarray = None
        self.phi_prev: ndarray = None

        # Angular flux vector
        self.psi: ndarray = None
        self.psi_interface: ndarray = None

        # Precursor vector
        self.precursors: ndarray = None

        # Source moment vector
        self.q_moments: ndarray = None

        # Angular operators
        self.M: ndarray = None
        self.D: ndarray = None

        # Angle Sets
        self.angle_sets: List[List[int]] = None

    def initialize(self, verbose: bool = True) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        verbose : bool, default True
        """
        self._check_inputs()

        # Initialize the materials
        self.initialize_materials()

        # Define number of moments and angles
        self.n_angles = self.compute_n_angles()
        self.n_moments = self.compute_n_moments()

        # Initialize the unknown information
        self.initialize_unknowns()

        # Initialize angular operators
        self.D = self.discrete_to_moment_matrix()
        self.M = self.moment_to_discrete_matrix()

        # Initialize boundaries
        self.initialize_bondaries()


    def execute(self,verbose: bool = True) -> None:
        """
        Execute the solver.

        Parameters
        ----------
        verbose : bool, default True
        """
        pass
