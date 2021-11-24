import numpy as np
from numpy import ndarray
from typing import List
import matplotlib.pyplot as plt

from pyPDEs.mesh import *
from pyPDEs.material import *
from pyPDEs.spatial_discretization import *
from pyPDEs.utilities import Vector, UnknownManager
from pyPDEs.utilities.boundaries import Boundary
from pyPDEs.utilities.quadratures import ProductQuadrature

from ..data_structures import *
from ..boundaries import *


class SteadyStateSolver:
    """
    Steady state multigroup neutron transport solver
    """

    from ._check_inputs import check_inputs
    from ._initialize_materials import initialize_materials
    from ._initialize_boundaries import (initialize_bondaries,
                                         _initialize_reflective_bc)
    from ._initialize_unknowns import initialize_unknowns

    from ._angular_operators import (discrete_to_moment_matrix,
                                     moment_to_discrete_matrix)
    from ._compute_anglesets import (initialize_angle_sets,
                                     create_sweep_ordering)

    from ._setsource import set_source
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

        # Quadrature object
        self.quadrature: ProductQuadrature = None
        self.harmonic_index_map: List[HarmonicIndex] = []
        self.angle_aggregation_type: str = 'octant'
        self.angle_sets: List[AngleSet] = []

        # Angular operators
        self.M: ndarray = None
        self.D: ndarray = None

        # Flux moment vectors
        self.phi: ndarray = None
        self.phi_prev: ndarray = None

        # Angular flux vector
        self.psi: ndarray = None
        self._fluds: ndarray = None

        # Precursor vector
        self.precursors: ndarray = None

        # Source moment vector
        self.q_moments: ndarray = None

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
        pw_change_prev = 1.0
        converged = False
        for k in range(self.max_iterations):
            self.q_moments *= 0.0
            self.set_source(self.q_moments)
            self.sweep()

            pw_change = np.linalg.norm(self.phi - self.phi_prev)
            self.phi_prev[:] = self.phi

            rho = np.sqrt(pw_change / pw_change_prev)
            if k == 0:
                rho = 0.0

            if pw_change < self.tolerance * (1.0 - rho):
                converged = True

            print(f'===== Iteration {k} =====')
            print(f'\tPoint-wise change:\t{pw_change:.3e}')
            print(f'\tSpectral radius est.:\t{rho:.3e}')
            print()

            if converged:
                print('***** CONVERGED *****')
                print()
                break

    def plot_flux_moment(self, moment_num: int, group_num: int) -> None:
        ell_m = self.harmonic_index_map[moment_num]
        if self.mesh.dim == 1:
            grid = [pt.z for pt in self.discretization.grid]

            plt.figure()
            plt.xlabel('z', fontsize=12)
            plt.ylabel(rf'$\phi_{{{ell_m.ell}, {group_num}}}^{ell_m.m}$(z)')
            plt.grid(True)

            plt.plot(grid, self.phi[moment_num][group_num])
            plt.show()

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

    def psi_upwind(self, cell_id: int, face_num: int,
                   angle_num: int) -> List[int]:
        """
        Get the group-wise upwind values for psi.

        Parameters
        ----------
        cell_id : int
        face_num : int
        angle_num : int

        Returns
        -------
        List[int]
        """
        face = self.mesh.cells[cell_id].faces[face_num]
        adj_cell = face.neighbor_id
        ass_face = face.associated_face
        return self._fluds[angle_num][adj_cell][ass_face]

    def psi_outflow(self, psi: float, cell_id: int, face_num: int,
                    angle_num: int, group_num: int) -> None:
        """
        Set the outflow psi after computing it in a sweep.

        Parameters
        ----------
        psi : float
        cell_id : int
        face_num : int
        angle_num : int
        group_num : int
        """
        n, c, f, g = angle_num, cell_id, face_num, group_num
        self._fluds[n][c][f][g] = psi

    def psi_boundary(self, bndry_id: int, cell_id: int,
                     face_num: int, angle_num: int) -> List[float]:
        """
        Get the boundary value for psi.

        Parameters
        ----------
        bndry_id : int
        cell_id : int
        face_num : int
        angle_num : int

        Returns
        -------
        List[float]
        """
        bc: Boundary = self.boundaries[bndry_id]
        if not isinstance(bc, ReflectiveBoundary):
            return bc.values
        else:
            n, c, f = angle_num, cell_id, face_num
            return bc.boundary_psi_incoming(n, c, f)
