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
    from ._classicrichardson import classic_richardson

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
        self.classic_richardson(verbose=verbose)

    def plot_flux_moment(self, ell: int, m: int, group_num: int) -> None:
        """
        Plot flux moments.

        Parameters
        ----------
        ell : int
        m: int
        group_num : int
        """
        moment_num = self.find_harmonic_index(ell, m)
        grid = self.discretization.grid

        # Plot 1D solutions
        if self.mesh.dim == 1:
            plt.figure()
            plt.title(f'Flux Moment\n$\ell$={ell}, $m$={m}', fontsize=12)
            plt.xlabel('z', fontsize=12)
            plt.ylabel('$\phi$', fontsize=12)

            grid = [pt.z for pt in grid]
            plt.plot(grid, self.phi[moment_num][group_num])
            plt.grid(True)

        # Plot 2D solutions
        elif self.mesh.dim == 2:
            plt.figure()
            plt.title(f'Flux Moment\n$\ell$={ell} $m$={m}', fontsize=12)
            plt.xlabel('X', fontsize=12)
            plt.ylabel('Y', fontsize=12)

            x = np.unique([p.x for p in grid])
            y = np.unique([p.y for p in grid])
            xx, yy = np.meshgrid(x, y)
            phi: ndarray = self.phi[moment_num][group_num]
            phi = phi.reshape(xx.shape)
            im = plt.pcolor(xx, yy, phi, cmap='jet', shading='auto',
                            vmin=0.0, vmax=phi.max())
            plt.colorbar(im)
        plt.tight_layout()

    def plot_angular_flux(self, angle_num: int, group_num: int) -> None:
        """
        Plot the angular flux for a specific angle and group.

        Parameters
        ----------
        angle_num : int
        group_num : int
        """
        omega = self.quadrature.omegas[angle_num]
        grid = self.discretization.grid
        if self.mesh.dim == 1:
            plt.figure()
            plt.title(f'Angular Flux\n$\mu = {omega.z:.3f}$', fontsize=12)
            plt.xlabel('z', fontsize=12)
            plt.ylabel(rf'$\psi_n$', fontsize=12)

            grid = [pt.z for pt in grid]
            plt.plot(grid, self.psi[angle_num][group_num])
            plt.grid(True)
        plt.tight_layout()

    def compute_piecewise_change(self) -> float:
        """
        Compute the point-wise change in phi.

        Returns
        -------
        float
        """
        # Loop over cells, moments, groups
        pw_change = 0.0
        for c in range(self.mesh.n_cells):
            for m in range(self.n_moments):
                phi_m = self.phi[m]
                phi_prev_m = self.phi_prev[m]

                for g in range(self.n_groups):

                    # Max scalar flux
                    abs_phi_m0 = abs(self.phi[0][g][c])
                    abs_phi_prev_m0 = abs(self.phi_prev[0][g][c])
                    max_phi = max(abs_phi_m0, abs_phi_prev_m0)

                    # Change in flux moment
                    dphi = abs(phi_m[g][c] - phi_prev_m[g][c])

                    # Compute max change
                    if max_phi >= 1.0e-16:
                        pw_change = max(dphi/max_phi, pw_change)
                    else:
                        pw_change = max(dphi, pw_change)
        return pw_change

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

    def find_harmonic_index(self, ell: int, m: int) -> int:
        """
        Get the moment number given a set of harmonic indices.

        Parameters
        ----------
        ell : int
        m : int

        Returns
        -------
        int
        """
        ell_m = HarmonicIndex(ell, m)
        for n in range(self.n_moments):
            if ell_m == self.harmonic_index_map[n]:
                return n
        raise ValueError(
            f'No harmonic index found for indices '
            f'ell, m = {ell}, {m}.')

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
