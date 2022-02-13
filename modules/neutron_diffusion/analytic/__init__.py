import os.path

import numpy as np

from numpy import ndarray
from scipy.linalg import eig

from sympy import sin, cos, pi
from sympy import integrate, lambdify, symbols
from sympy import Matrix, Expr

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes

from typing import List, Tuple

from pyPDEs.material import CrossSections
from .modes import AlphaMode

Eigenproblem = Tuple[ndarray, ndarray, ndarray]

__all__ = ['AnalyticSolution', 'load']


class AnalyticSolution:
    """
    Analytic solution for multi-group neutron diffusion.

    The analytic solution is an alpha-eigenfunction expansion.
    The alpha eigenfunction is doubly indexed for spatial and
    energy profile. Currently only
    """
    def __init__(self, xs: CrossSections, ics: Matrix, rf: float,
                 coord_sys: str, tolerance: float = 1.0e-8,
                 max_n_modes: int = int(5.0e3)) -> None:
        self.xs: CrossSections = xs
        self.initial_conditions: Matrix = ics

        self.rf: float = rf
        self.coord_sys: str = coord_sys

        self.tolerance: float = tolerance
        self.max_n_modes: int = max_n_modes

        self.modes: List[AlphaMode] = []

        self._rhs: Expr = None
        self._lhs: List[Expr] = None

        self._eigval_order: List[int] = []
        self._amplitude_order: List[int] = []

    @property
    def n_groups(self) -> int:
        return self.xs.n_groups

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def varphi_n(self) -> Expr:
        r, n = symbols('r, n')
        if self.coord_sys == 'cartesian':
            coeff = 0.5 * (2*n - 1) * pi / self.rf
            return cos(coeff * r)
        else:
            coeff = n * pi / self.rf
            return sin(coeff * r) / (coeff * r)

    @property
    def dominant_mode(self) -> int:
        return self._amplitude_order[0]

    def execute(self) -> None:
        """
        Fit an alpha-eigenfuction expansion to the initial conditions.
        """
        self._check_inputs()

        # Print header
        msg = "===== Computing alpha-eigenfunction expansion ====="
        msg = "\n".join(["", "="*len(msg), msg, "="*len(msg)])
        print(msg)

        # Precompute spatial integrals. Results are function of n
        import time
        print(f"Starting integrations...")
        t_start = time.time()
        self._rhs = self._integrate_volume(self.varphi_n**2)
        self._lhs = []
        for g in range(self.n_groups):
            ic = self.initial_conditions[g]
            self._lhs += [self._integrate_volume(self.varphi_n * ic)]
        print(f"Integrations took {time.time() - t_start:.3g} sec\n")

        # Defne the reference grid
        n_pts = int(1.0e4)
        dr = self.rf / n_pts
        r = np.linspace(0.5*dr, self.rf - 0.5*dr, n_pts)

        # Compute initial condition on grid
        ic = self.compute_initial_values(r)
        fit = np.zeros(len(r) * self.n_groups)

        self.modes.clear()
        mode_num, diff = 0, 1.0
        while diff > self.tolerance and mode_num < self.max_n_modes:
            decomp = self.solve_alpha_eigenproblem(mode_num)
            alphas, f_adjoint, f = decomp
            b = self.compute_amplitudes(mode_num, f, f_adjoint)

            # Construct n, m eigenfunctions
            for g in range(self.n_groups):
                mode = AlphaMode(self, mode_num, g, alphas[g],
                                 b[g], f[:, g], f_adjoint[:, g])
                self.modes.append(mode)

                fit += mode.evaluate_mode(r)
                diff = np.linalg.norm(fit - ic)

            mode_num += 1

        # Sort by eigenvalue
        k = lambda i: abs(self.modes[i].alpha.real)
        self._eigval_order = sorted(list(range(self.n_modes)), key=k)
        self.modes = [self.modes[i] for i in self._eigval_order]

        # Define amplitude sorting
        k = lambda i: abs(self.modes[i].b.real)
        self._amplitude_order = sorted(list(range(self.n_modes)),
                                       key=k, reverse=True)


        print(f'Dominant Mode:\t{self.dominant_mode}')
        print(f'# of Modes:\t{len(self.modes)}')
        print(f'IC Fit Error:\t{diff:.4e}\n')

    def evaluate_expansion(self, r: ndarray, t: ndarray) -> ndarray:
        """
        Evaluate the alpha-eigenfunction expansion.

        Parameters
        ----------
        r : ndarray
            The spatial points to evaluate the expansion at.
        t : ndarray
            The time points to evaluate the expansion at.

        Returns
        -------
        ndarray (len(t), len(r) * n_groups)
        """
        if t is None:
            t = np.zeros(1)
        elif isinstance(t, float):
            t = np.array([t])
        t = np.array(t)

        phi = np.zeros((len(t), len(r)*self.n_groups), dtype=complex)
        for mode in self.modes:
            phi += mode.evaluate_mode(r, t)
        return phi.real if len(t) > 1 else phi.real.ravel()

    def get_mode(self, n: int, m: int = None,
                 method: str = 'nm') -> AlphaMode:
        """
        Return the specified alpha-eigenfunction object.

        Parameters
        ----------
        n : int
            The spatial index.
        m : int
            The energy index.
        method : {'nm', 'eig', 'amp'}
            Whether to get mode n, m, the mode with
            the n'th largest eigenvalue, or the n'th largest
            amplitude.

        Returns
        -------
        AlphaMode
        """
        if method == 'nm':
            for mode in self.modes:
                if mode.n == n + 1 and mode.m == m:
                    return mode
        elif method == 'eig':
            return self.modes[n]
        elif method == 'amp':
            return self.modes[self._amplitude_order[n]]

    def plot_mode(self, r: ndarray, n: int,
                  m: int = 0, method: str = 'nm') -> None:
        """
        Plot the specified alpha-eigenfunction.

        Parameters
        ----------
        r : ndarray
            The grid to plot the eigenfunction on.
        n : int
            The spatial index.
        m : int
            The energy index.
        method : {'nm', 'eig', 'amp'}
            Whether to get mode n, m, the mode with
            the n'th largest eigenvalue, or the n'th largest
            amplitude.
        """
        self.get_mode(n, m, method).plot_mode(r)

    def plot_expansion(self, r: ndarray, t: float) -> None:
        r = np.array(r)
        phi = self.evaluate_expansion(r, t)

        # Initialize the figure
        fig: Figure = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_title(f"Time = {t:.3e} $\\mu$s")
        ax.set_xlabel("r (cm)", fontsize=12)
        ax.set_ylabel(r"$\phi_g(r)$", fontsize=12)

        # Plot the profiles
        for g in range(self.n_groups):
            ax.plot(phi[g::self.n_groups], label=f'Group {g}')
        ax.legend()
        ax.grid(True)

    def find_mode_index_from_alpha(self, alpha: float) -> int:
        """
        Find the nearest mode to the provided alpha.

        Parameters
        ----------
        alpha : float

        Returns
        -------
        AlphaMode
        """
        diff = [abs(mode.alpha.real - alpha) for mode in self.modes]
        return np.argmin(diff)

    def solve_alpha_eigenproblem(self, mode_num: int) -> Eigenproblem:
        """
        Solve the 0D alpha-eigenproblem for a given mode.

        Parameters
        ----------
        mode_num : int
            The mode number. This is used to compute the buckling term.

        Returns
        -------
        ndarray (n_groups,)
            The alpha-eigenvalues.
        ndarray (n_groups,) * 2
            The adjoint energy-weight eigenvectors.
        ndarray (n_groups,) * 2
            The energy-weight eigenvectors.
        """
        if self.coord_sys == 'cartesian':
            Bn = 0.5 * (mode_num + 1) * np.pi / self.rf
        else:
            Bn = (mode_num + 1) * np.pi / self.rf

        A = np.zeros((self.n_groups,) * 2)
        for g in range(self.n_groups):
            A[g][g] -= Bn**2 * self.xs.D[g] + self.xs.sigma_t[g]
            for gp in range(self.n_groups):
                A[g][gp] += \
                    self.xs.chi[g] * self.xs.nu_sigma_f[gp] + \
                    self.xs.transfer_matrix[0][gp][g]
            A[g] *= self.xs.velocity[g]
        return eig(A, left=True, right=True)

    def compute_amplitudes(self, mode_num: int, f: ndarray,
                           f_adjoint: ndarray) -> ndarray:
        """
        Compute the mode amplitudes.

        This routine projects the adjoint modes onto the
        initial condition definition to compute the modes.
        The inner product used is an integration over the
        domain volume and sum over groups.

        Parameters
        ----------
        mode_num : int
            The mode number the amplitudes are being computed for.
        f : ndarray (n_groups,) * 2
            The energy-weight eigenvectors.
        f_adjoint : ndarray (n_groups,) * 2
            The adjoint energy-weight eigenvectors.

        Returns
        -------
        ndarray (n_groups,)
            The n_groups mode amplitudes for mode `mode_num`.
        """
        n = symbols('n')
        rhs = self._rhs.subs(n, mode_num + 1).evalf()
        lhs = [lhs_g.subs(n, mode_num + 1).evalf() for lhs_g in self._lhs]

        b = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            b[g] = np.dot(lhs, f_adjoint[:, g])
            b[g] /= rhs * np.dot(f_adjoint[:, g], f[:, g])
        return b

    def compute_initial_values(self, points: ndarray) -> ndarray:
        """
        Compute the initial values at `points`.

        Parameters
        ----------
        points : ndarray
            The points to evaluate the initial conditions at.

        Returns
        -------
        ndarray (len(points) * n_groups)
        """
        r = symbols('r')
        vec = np.zeros(len(points) * self.n_groups)
        for g in range(self.n_groups):
            ic = lambdify(r, self.initial_conditions[g])
            vec[g::self.n_groups] = np.ravel(ic(points))
        return vec

    def _integrate_volume(self, f: Expr) -> Expr:
        """
        Perform a definite integral over a volume.

        Parameters
        ----------
        f : Expr
            The function to integrate.

        Returns
        --------
        Expr
        """
        r = symbols('r')
        if self.coord_sys == 'cartesian':
            return integrate(f, (r, 0.0, self.rf))
        else:
            jac = 4.0*pi*r**2
            return integrate(f * jac, (r, 0.0, self.rf))

    def _check_inputs(self) -> None:
        if self.rf is None:
            raise ValueError('No system width was provided.')
        if self.rf <= 0.0:
            raise ValueError('The system width must be greater than zero.')

        cs = ['cartesian', 'spherical']
        if self.coord_sys not in cs:
            raise ValueError(
                f'Only {cs[0].lower()} and {cs[1].lower()} '
                f'coodinate systems are allowed.')

        if self.xs is None:
            raise ValueError('No cross sections were provided.')

        if self.initial_conditions is None:
            raise ValueError('No initial conditions were provided.')
        if not isinstance(self.initial_conditions, Matrix):
            raise TypeError('ICs must be of type `sympy.Matrix`.')
        if len(self.initial_conditions) != self.n_groups:
            raise ValueError('There must be n_groups ICs.')

    def save(self, path: str) -> None:
        """
        Save the analytic solution object via pickle.

        Parameters
        ----------
        path : str
            The path to the destination.
        """
        import pickle
        dir_path = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        if '.' in path:
            if len(path.split('.')) != 2:
                raise ValueError(
                    'Invalid filepath with multiple extensions.')
            path = '.'.join([path.split('.')[0], 'obj'])
        else:
            path = '.'.join([path, 'obj'])

        with open(path, mode='wb') as f:
            pickle.dump(self, f)


def load(path: str) -> AnalyticSolution:
    import pickle
    if '.obj' not in path:
        raise ValueError('Only .obj files can be loaded.')
    with open(path, mode='rb') as f:
        return pickle.load(f)
