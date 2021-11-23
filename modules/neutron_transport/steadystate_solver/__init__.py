import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import Vector, UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature

from ..data_structures import FLUDS, HarmonicIndex, AngleSet
from ..boundaries import *


class SteadyStateSolver:
    """
    Steady state multigroup neutron transport solver
    """

    from ._check_inputs import check_inputs
    from ._initialize_materials import initialize_materials
    from ._initialize_boundaries import initialize_bondaries
    from ._initialize_unknowns import initialize_unknowns

    from ._angular_operators import (discrete_to_moment_matrix,
                                     moment_to_discrete_matrix)
    from ._compute_anglesets import create_angle_sets

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
        self.angle_aggregation_type: str = 'octant'
        self.angle_sets: List[AngleSet] = []

        # Unknown managers
        self.phi_uk_man: UnknownManager = None
        self.psi_uk_man: UnknownManager = None

        # Flux moment vectors
        self.phi: ndarray = None
        self.phi_prev: ndarray = None

        # Angular flux vector
        self.psi: ndarray = None

        # Precursor vector
        self.precursors: ndarray = None

        # Source moment vector
        self.q_moments: ndarray = None

        # Angular operators
        self.M: ndarray = None
        self.D: ndarray = None

    def initialize(self, verbose: bool = True) -> None:
        """
        Initialize the solver.

        Parameters
        ----------
        verbose : bool, default True
        """
        self.check_inputs()

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

        # Compute angle sets
        self.initialize_angle_sets()


    def execute(self, verbose: bool = True) -> None:
        """
        Execute the solver.

        Parameters
        ----------
        verbose : bool, default True
        """
        pass

    def compute_n_angles(self) -> int:
        """
        Compute the number of angles.

        Returns
        -------
        int
        """
        return len(self.quadrature.abscissae)

    def compute_n_moments(self) -> int:
        """
        Compute the number of moments.

        Returns
        -------
        int
        """
        self.create_harmonic_indices()
        return len(self.harmonic_index_map)

    def create_harmonic_indices(self) -> None:
        """
        Generate the harmonic index ordering.
        """
        self.harmonic_index_map.clear()
        if self.mesh.dim == 1:
            for ell in range(self.scattering_order + 1):
                self.harmonic_index_map.append(HarmonicIndex(ell, 0))
        elif self.mesh.dim == 2:
            for ell in range(self.scattering_order + 1):
                for m in range(-ell, ell + 1, 2):
                    if ell == 0 or m != 0:
                        self.harmonic_index_map.append(HarmonicIndex(ell, m))
        else:
            for ell in range(self.scattering_order + 1):
                for m in range(-ell, ell + 1):
                    self.harmonic_index_map.append(HarmonicIndex(ell, m))

    def initialize_angle_sets(self) -> None:
        """
        Initialize the angle sets for the problem.
        """
        self.angle_sets.clear()

        # Octant aggregation
        if self.angle_aggregation_type == 'octant':

            # 1D hemisphere aggregation
            if self.mesh.dim:
                sweep_ordering = list(range(self.mesh.n_cells))

                # Top hemisphere
                as_top = AngleSet()
                as_top.sweep_ordering = sweep_ordering
                for i, omega in enumerate(self.quadrature.omegas):
                    if omega.z > 0.0:
                        as_top.angles.append(i)
                self.angle_sets.append(as_top)

                # Bottom hemisphere
                as_bot = AngleSet()
                as_bot.sweep_ordering = sweep_ordering[::-1]
                for i, omega in enumerate(self.quadrature.omegas):
                    if omega.z < 0.0:
                        as_bot.angles.append(i)
                self.angle_sets.append(as_bot)

        # Initialize FLUDs
        for angle_set in self.angle_sets:
            angle_set: AngleSet = angle_set
            angle_set.fluds = FLUDS()
            angle_set.fluds.psi = \
                np.zeros((self.mesh.n_cells, 2*self.mesh.dim,
                          len(angle_set.angles), self.n_groups))
