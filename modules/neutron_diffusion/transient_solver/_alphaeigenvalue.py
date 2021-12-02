import numpy as np
from numpy import ndarray
from numpy.linalg import inv
from scipy.linalg import eig
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from . import TransientSolver


AEE = 'AlphaEigenfunctionExpansion'


class AlphaEigenfunctionExpansion:
    def __init__(self, alphas: ndarray, eigvecs: ndarray,
                 adjoint_eigvecs: ndarray, amplitudes: ndarray) -> None:
        self._alphas: ndarray = alphas
        self._eigvecs: ndarray = eigvecs
        self._adj_eigvecs: ndarray = adjoint_eigvecs
        self._b: ndarray = amplitudes

    @property
    def n_modes(self) -> int:
        return len(self._alphas)

    @property
    def nonzero_amplitudes(self) -> int:
        count = 0
        for m in range(self.n_modes):
            if self._b[m].real > 1.0e-8:
                count += 1
        return count

    @property
    def alphas(self) -> ndarray:
        return self._alphas

    @property
    def eigenvectors(self) -> ndarray:
        return self._eigvecs

    @property
    def adjoint_eigenvectors(self) -> ndarray:
        return self._adj_eigvecs

    @property
    def amplitudes(self) -> ndarray:
        return self._b

    def evaluate_expansion(self, times: List[float]) -> ndarray:
        if not isinstance(times, list):
            times = [times]

        n_dofs = self._eigvecs.shape[0]
        phi = np.zeros((n_dofs, len(times)), dtype=complex)
        for t, time in enumerate(times):
            for m in range(self.n_modes):
                attn = np.exp(self._alphas[m] * time)
                phi[:, t] += self._b[m] * self._eigvecs[:, m] * attn
        return phi.real


def solve_alpha_eigenproblem(self: 'TransientSolver',
                             tolerance: float = 1.0e-6,
                             max_modes: int = -1) -> AEE:
    # Compute relevant matrices
    A = -self.assemble_matrix().todense()
    M = self.mass_matrix().todense()

    # Check max modes
    n_dofs = A.shape[0]
    if max_modes < 0 or max_modes > n_dofs:
        max_modes = n_dofs

    # Solve the eigenvalue problem
    vals, vecs_l, vecs_r = eig(inv(M)@A, left=True, right=True)

    # Sort by eigenvalue
    idx = np.argsort(vals)[::-1] # greatest to least
    vals = vals[idx]
    vecs_r = vecs_r[:, idx]
    vecs_l = vecs_l[:, idx]

    # Compute initial condition
    phi_ic = np.zeros(self.phi.shape)
    grid = [node.z for node in self.discretization.grid]
    for g in range(self.n_groups):
        if not callable(self.initial_conditions[g]):
            raise AssertionError('Initial conditions must be callable.')
        ic = self.initial_conditions[g]
        phi_ic[g::self.n_groups] = ic(np.array(grid))

    # Compute inner produces
    vecs_l_star = vecs_l.conj().T
    A = vecs_l_star @ vecs_r
    b = vecs_l_star @ phi_ic.reshape(-1, 1)
    b = np.linalg.solve(A, b).ravel()

    # Enforce positive amplitudes
    for m in range(len(vals)):
        if b[m] < 0.0:
            b[m] *= -1.0
            vecs_r[:, m] *= -1.0
            vecs_l[:, m] *= -1.0

    # Eliminate effective zero amplitudes
    nonzero_map = b > 1.0e-12
    vals = vals[nonzero_map]
    vecs_r = vecs_r[:, nonzero_map]
    vecs_l = vecs_l[:, nonzero_map]
    b = b[nonzero_map]

    # Reset max modes
    max_modes = min(max_modes, len(vals))

    # Truncate, if possible
    fit = np.zeros(phi_ic.shape)
    for n in range(max_modes):
        fit += b[n] * vecs_r[:, n]

        diff = np.linalg.norm(phi_ic - fit)
        if diff < tolerance or n + 1 == max_modes:
            vals = vals[:n+1]
            vecs_r = vecs_r[:, :n+1]
            vecs_l = vecs_l[:, :n+1]
            b = b[:n+1]
            break

    # Print summary
    print(f'\nNumber of Modes:\t{len(vals)}')
    print(f'Error in IC Fit:\t{diff:.3e}\n')

    return AlphaEigenfunctionExpansion(vals, vecs_r, vecs_l, b)


