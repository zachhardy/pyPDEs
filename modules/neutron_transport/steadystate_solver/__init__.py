import numpy as np
from numpy import ndarray

from pyPDEs.mesh import Mesh
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import Vector, UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature

from ..boundaries import *


class HarmonicIndex:
    """
    Structure for spherical harmonic indices.
    """

    def __init__(self, ell: int, m: int) -> None:
        self.ell: int = ell
        self.m: int = m

    def __eq__(self, other: 'HarmonicIndex') -> bool:
        return self.ell == other.ell and self.m == other.m


class AngleSet:
    """
    Implementation for an angle set.

    An angle set is defined as a collection of angles that
    share a sweep ordering.
    """
    def __init__(self) -> None:
        self.angles: List[int] = []
        self.sweep_ordering: List[int] = []


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

    from ._setsource import set_source
    from ._source_iterations import source_iterations
    from ._sweep import sweep

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
        self.psi_outflow: ndarray = None

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

    def create_angle_sets(self) -> None:
        """
        Create the angle sets.
        """
        # Clear current angle sets
        self.angle_sets.clear()

        # 1D problems
        if self.mesh.dim == 1:
            sweep_order = list(range(self.mesh.n_cells))

            # Rightward directions
            angle_set = AngleSet()
            angle_set.sweep_ordering = sweep_order
            for i in range(self.n_angles):
                omega = self.quadrature.omegas[i]
                if omega.z > 0.0:
                    angle_set.angles.append(i)
            self.angle_sets.append(angle_set)

            # Leftward directions
            angle_set = AngleSet()
            angle_set.sweep_ordering = sweep_order[::-1]
            for i in range(self.n_angles):
                omega = self.quadrature.omegas[i]
                if omega.z < 0.0:
                    angle_set.angles.append(i)
            self.angle_sets.append(angle_set)
