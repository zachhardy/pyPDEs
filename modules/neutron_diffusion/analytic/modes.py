import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from scipy.linalg import eig
from matplotlib.pyplot import Figure, Axes

from sympy import symbols


from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from . import AnalyticSolution

Function = Callable[[ndarray], ndarray]


class AlphaEigenfunction:
    """
    Alpha-eigenfunction for a given spatio-energy mode.
    """

    def __init__(self, expansion: 'AnalyticSolution',
                 mode_num: int, group_num: int,
                 alpha: complex, b: complex,
                 f: ndarray, f_adjoint: ndarray) -> None:
        self.n: int = mode_num
        self.m: int = group_num

        self.alpha: complex = alpha
        self.b: complex = b
        self.f: ndarray = f
        self.f_adjoint: ndarray = f_adjoint

        self._r_f: float = expansion.rf
        self._n_groups: int = expansion.n_groups
        self._coord_sys: str = expansion.coord_sys

    def evaluate_eigenfunction(self, r: ndarray, t: ndarray = None,
                               with_amplitudes: bool = True) -> ndarray:
        """
        Evaluate the eigenfunction on a spatiotemporal grid.

        Parameters
        ----------
        r : ndarray
            The spatial points to evaluate the eigenfunction at.
        t : ndarray, default None
            The time points to evaluate the eigenfunction at.
            Default behavior is to evaluate the eigenfunction
            at the initial condition.
        with_amplitudes : bool, default True
            A flag to scale the eigenfuction by the computed amplitude
            or not.

        Returns
        -------
        ndarray (len(t), len(r) * n_groups)
        """
        if t is None:
            t = np.zeros(1)
        elif isinstance(t, float):
            t = np.array([t])
        t = np.array(t)

        # Evaluate varphi
        varphi = self.varphi(r)
        b = self.b if with_amplitudes else 1.0
        phi = np.zeros((len(t), len(r)*self._n_groups), dtype=complex)
        for n, tn in enumerate(t):
            for g in range(self._n_groups):
                phi[n, g::self._n_groups] = self.f[g] * varphi
            phi[n, :] *= b * np.exp(self.alpha * tn)
        return phi.real if len(t) > 1 else phi.real.ravel()

    def varphi(self, r: ndarray) -> ndarray:
        """
        Evaluate the stored function for varphi.

        Parameters
        ----------
        r : ndarray
            The points to evaluate varphi at.

        Returns
        -------
        ndarray (len(r),)
        """
        r = np.array(r)
        if self._coord_sys == 'cartesian':
            coeff = 0.5 * (2 * self.n - 1) * np.pi / self._r_f
            return np.cos(coeff * r)
        else:
            coeff = self.n * np.pi / self._r_f
            return np.sin(coeff * r) / (coeff * r)

    def plot_eigenfunction(self, r: ndarray) -> None:
        """
        Plot the eigenfunction profile.

        Parameters
        ----------
        r : ndarray
            The grid to plot the eigenfunction on.
        """
        r = np.array(r)
        phi = self.evaluate_eigenfunction(r, with_amplitudes=False)
        phi /= np.linalg.norm(phi)
        phi *= -1.0 if phi[0] < 0 else phi

        # Initialize the figure
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('r', fontsize=12)
        ax.set_ylabel(r'$f_{nm,g}$ $\varphi_{n}(r)$', fontsize=12)
        title = f'Alpha Mode n={self.n-1}, m={self.m}\n' \
                f'$\\alpha$ = {self.alpha.real:.3e}{self.alpha.imag:+.5g}j'
        fig.suptitle(title, fontsize=12)

        # Plot the profiles
        for g in range(self._n_groups):
            ax.plot(phi[g::self._n_groups], label=f'Group {g}')
        ax.legend()
        ax.grid(True)
