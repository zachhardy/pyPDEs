import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.linalg import eig
from typing import List

from pyPDEs.spatial_discretization import *
from ..steadystate_solver import SteadyStateSolver


class AlphaEigenvalueSolver(SteadyStateSolver):
    """
    Implementation of an analytic alpha-eigenvalue solver.
    """
    def __init__(self) -> None:
        super().__init__()

        self._alphas: ndarray = None
        self._modes: ndarray = None
        self._adjoint_modes: ndarray = None
        self._b: ndarray = None

    @property
    def n_modes(self) -> int:
        return len(self._alphas)

    @property
    def alphas(self) -> ndarray:
        return self._alphas

    @property
    def modes(self) -> ndarray:
        return self._modes

    @property
    def adjoint_modes(self) -> ndarray:
        return self._adjoint_modes

    @property
    def amplitudes(self) -> ndarray:
        return self._b

    def eigendecomposition(self, ic: List[callable],
                           verbose: bool = False) -> None:
        A = self.assemble_eigensystem()
        evals, evecs_l, evecs_r = eig(A, left=True)

        idx = np.argsort(abs(evals))
        evals = evals[idx]
        evecs_r = evecs_r[:, idx]
        evecs_l = evecs_l[:, idx]

        # Compute initial condition
        phi_ic = np.zeros(self.phi.shape)
        z = np.array([p.z for p in self.discretization.grid])
        for g in range(self.n_groups):
            if not callable(ic[g]):
                raise AssertionError('IC functions must be callable.')
            phi_ic[g::self.n_groups] = ic[g](z)

        # Compute inner products
        evecs_l_star = evecs_l.conj().T
        A = evecs_l_star @ evecs_r
        b = evecs_l_star @ phi_ic.reshape(-1, 1)
        b = np.linalg.solve(A, b).ravel()

        # Enforce positive amplitudes
        for m in range(len(evals)):
            if b[m] < 0.0:
                b[m] *= -1.0
                evecs_r[:, m] *= -1.0
                evecs_l[:, m] *= -1.0

        # Eliminate zero amplitudes
        nonzero_map = b > 1.0e-12
        vals = evals[nonzero_map]
        evecs_r = evecs_r[:, nonzero_map]
        evecs_l = evecs_l[:, nonzero_map]
        b = b[nonzero_map]

        self.phi = evecs_r @ b
        diff = np.linalg.norm(phi_ic - self.phi)

        if verbose:
            print(f'\nDifference in IC Fit\t{diff:.3e}\n')

        # Set properties
        self._alphas = evals
        self._modes = evecs_r
        self._adjoint_modes = evecs_l
        self._b = b

    def assemble_eigensystem(self) -> ndarray:
        """
        Assemble the alpha-eigenvalu problem matrix.

        Returns
        -------
        ndarray
        """
        fv: FiniteVolume = self.discretization
        A = self.assemble_matrix().todense()
        for cell in self.mesh.cells:
            xs_id = self.matid_to_xs_map[cell.material_id]
            xs = self.material_xs[xs_id]
            for g in range(self.n_groups):
                dof = fv.map_dof(cell, 0, self.phi_uk_man, 0, g)
                A[dof] *= xs.velocity[g] / cell.volume
        return -A
