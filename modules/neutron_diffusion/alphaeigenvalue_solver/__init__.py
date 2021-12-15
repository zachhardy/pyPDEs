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
        self.initial_condition: List[callable] = None

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

    @property
    def dominant_mode(self) -> int:
        return np.argmax(self._b.real)

    def eigendecomposition(self, verbose: bool = False) -> None:
        A = self.assemble_eigensystem()
        evals, evecs_l, evecs_r = eig(A, left=True)

        idx = np.argsort(abs(evals))
        evals = evals[idx]
        evecs_r = evecs_r[:, idx]
        evecs_l = evecs_l[:, idx]

        self._alphas = evals
        self._modes = evecs_r
        self._adjoint_modes = evecs_l

        if self.initial_condition is not None:
            self.fit_to_initial_condition(verbose)

    def fit_to_initial_condition(self, verbose: bool) -> None:
        # Evaluate initial condition
        phi = np.zeros(self.phi.shape)
        z = np.array([p.z for p in self.discretization.grid])
        for g in range(self.n_groups):
            if not callable(self.initial_condition[g]):
                raise AssertionError('IC functions must be callable.')
            phi[g::self.n_groups] = self.initial_condition[g](z)

        # Compute amplitudes
        evecs_l_star = self._adjoint_modes.conj().T
        A = evecs_l_star @ self._modes
        b = evecs_l_star @ phi.reshape(-1, 1)
        self._b = np.linalg.solve(A, b).ravel()

        # Enforce positive amplitudes
        for m in range(self.n_modes):
            if self._b[m] < 0.0:
                self._b[m] *= -1.0
                self._modes[:, m] *= -1.0
                self._adjoint_modes[:, m] *= -1.0

        # Compute the initial condition
        self.phi = self._modes @ self._b
        diff = np.linalg.norm(phi - self.phi)

        if verbose:
            print(f'\nDifference in IC Fit\t{diff:.3e}\n')

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
