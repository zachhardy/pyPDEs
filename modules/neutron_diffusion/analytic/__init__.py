import os.path

import numpy as np

from numpy import ndarray
from scipy.linalg import eig

from sympy import sin, cos, pi
from sympy import integrate, lambdify, symbols
from sympy import Matrix, Expr

import matplotlib.pyplot as plt

from typing import List, Callable, Tuple

from pyPDEs.material import CrossSections
from .modes import AlphaEigenfunction

Eigenproblem = Tuple[ndarray, ndarray, ndarray]


class AnalyticSolution:
    """Analytic solution for multi-group neutron diffusion.

    The analytic solution is an alpha-eigenfunction expansion.
    The alpha eigenfunction is doubly indexed for spatial and
    energy profile. Currently only
    """
    def __init__(self, xs: CrossSections, ics: Matrix, rf: float,
                 coord_sys: str, tolerance: float = 1.0e-8,
                 max_n_modes: int = int(5.0e3)) -> None:
        """Constructor.
        """
        self.xs: CrossSections = xs
        self.initial_conditions: Matrix = ics

        self.rf: float = rf
        self.coord_sys: str = coord_sys

        self.tolerance: float = tolerance
        self.max_n_modes: int = max_n_modes

        self.modes: List[AlphaEigenfunction] = []

        self._rhs: Expr = None
        self._lhs: List[Expr] = None

    @property
    def n_groups(self) -> int:
        return self.xs.n_groups

    @property
    def varphi_n(self) -> Expr:
        r, n = symbols("r, n")
        if self.coord_sys == "CARTESIAN":
            coeff = 0.5 * (2*n - 1) * pi / self.rf
            return cos(coeff * r)
        else:
            coeff = n * pi / self.rf
            return sin(coeff * r) / (coeff * r)

    def execute(self) -> None:
        """Fit an alpha-eigenfuction expansion to the initial conditions.
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
            mode_num += 1
            decomp = self.solve_alpha_eigenproblem(mode_num)
            alphas, f_adjoint, f = decomp
            b = self.compute_amplitudes(mode_num, f, f_adjoint)

            # Construct n, m eigenfunctions
            for g in range(self.n_groups):
                mode = AlphaEigenfunction(self, mode_num, g, alphas[g],
                                          b[g], f[:, g], f_adjoint[:, g])
                self.modes.append(mode)

                fit += mode.evaluate_eigenfunction(r)
                diff = np.linalg.norm(fit - ic)

        print(f"# of Modes:\t{len(self.modes)}")
        print(f"IC Fit Error:\t{diff:.4e}\n")

    def evaluate_expansion(self, r: ndarray, t: ndarray) -> ndarray:
        """Evaluate the alpha-eigenfunction expansion.

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
            phi += mode.evaluate_eigenfunction(r, t)
        return phi.real if len(t) > 1 else phi.real.ravel()

    def get_eigenfunction(self, n: int, m: int) -> AlphaEigenfunction:
        """Return the specified alpha-eigenfunction object.

        Parameters
        ----------
        n : int
            The spatial index.
        m : int
            The energy index.

        Returns
        -------
        AlphaEigenfunction
        """
        for mode in self.modes:
            if mode.n == n + 1 and mode.m == m:
                return mode

    def plot_eigenfunction(self, n: int, m: int, r: ndarray) -> None:
        """Plot the specified alpha-eigenfunction.

        Parameters
        ----------
        n : int
            The spatial index.
        m : int
            The energy index.
        r : ndarray
            The grid to plot the eigenfunction on.
        """
        self.get_eigenfunction(n, m).plot_eigenfunction(r)

    def solve_alpha_eigenproblem(self, mode_num: int) -> Eigenproblem:
        """Solve the 0D alpha-eigenproblem for a given mode.

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
        if self.coord_sys == "CARTESIAN":
            Bn = 0.5 * mode_num * np.pi / self.rf
        else:
            Bn = mode_num * np.pi / self.rf

        A = np.zeros((self.n_groups,) * 2)
        for g in range(self.n_groups):
            A[g][g] -= Bn**2 * self.xs.D[g] + self.xs.sigma_t[g]
            for gp in range(self.n_groups):
                A[g][gp] += \
                    self.xs.chi[g] * self.xs.nu_sigma_f[gp] + \
                    self.xs.transfer_matrix[gp][g]
            A[g] *= self.xs.velocity[g]
        return eig(A, left=True, right=True)

    def compute_amplitudes(self, mode_num: int, f: ndarray,
                           f_adjoint: ndarray) -> ndarray:
        """Compute the mode amplitudes.

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
        n = symbols("n")
        rhs = self._rhs.subs(n, mode_num).evalf()
        lhs = [lhs_g.subs(n, mode_num).evalf() for lhs_g in self._lhs]

        b = np.zeros(self.n_groups)
        for g in range(self.n_groups):
            b[g] = np.dot(lhs, f_adjoint[:, g])
            b[g] /= rhs * np.dot(f_adjoint[:, g], f[:, g])
        return b

    def compute_initial_values(self, points: ndarray) -> ndarray:
        """Compute the initial values at `points`.

        Parameters
        ----------
        points : ndarray
            The points to evaluate the initial conditions at.

        Returns
        -------
        ndarray (len(points) * n_groups)
        """
        r = symbols("r")
        vec = np.zeros(len(points) * self.n_groups)
        for g in range(self.n_groups):
            ic = lambdify(r, self.initial_conditions[g])
            vec[g::self.n_groups] = np.ravel(ic(points))
        return vec

    def _integrate_volume(self, f: Expr) -> Expr:
        """Perform a definite integral over a volume.

        Parameters
        ----------
        f : Expr
            The function to integrate.

        Returns
        --------
        Expr
        """
        r = symbols("r")
        if self.coord_sys == "CARTESIAN":
            return integrate(f, (r, 0.0, self.rf))
        else:
            jac = 4.0*pi*r**2
            return integrate(f * jac, (r, 0.0, self.rf))

    def _check_inputs(self) -> None:
        if self.rf is None:
            raise ValueError("No system width was provided.")
        if self.rf <= 0.0:
            raise ValueError("The system width must be greater than zero.")

        cs = ["CARTESIAN", "SPHERICAL"]
        if self.coord_sys not in cs:
            raise ValueError(
                f"Only {cs[0].lower()} and {cs[1].lower()} "
                f"coodinate systems are allowed.")

        if self.xs is None:
            raise ValueError("No cross sections were provided.")

        if self.initial_conditions is None:
            raise ValueError("No initial conditions were provided.")
        if not isinstance(self.initial_conditions, Matrix):
            raise TypeError("ICs must be of type `sympy.Matrix`.")
        if len(self.initial_conditions) != self.n_groups:
            raise ValueError("There must be n_groups ICs.")

    def save(self, path: str) -> None:
        """Save the analytic solution object via pickle.

        Parameters
        ----------
        path : str
            The path to the destination.
        """
        import pickle
        dir_path = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        if "." in path:
            if len(path.split(".")) != 2:
                raise ValueError(
                    "Invalid filepath with multiple extensions.")
            path = ".".join([path.split(".")[0], "obj"])
        else:
            path = ".".join([path, "obj"])

        with open(path, mode="wb") as f:
            pickle.dump(self, f)


def load(path: str) -> AnalyticSolution:
    import pickle
    if ".obj" not in path:
        raise ValueError("Only .obj files can be loaded.")
    with open(path, mode="rb") as f:
        return pickle.load(f)


__all__ = ["AnalyticSolution", "load"]
