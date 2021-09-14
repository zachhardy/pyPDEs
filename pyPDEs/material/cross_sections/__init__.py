import numpy as np

from numpy import ndarray
from typing import Callable

from ..material_property import MaterialProperty

XSFunc = Callable[[int, float], float]


class CrossSections(MaterialProperty):
    """Neutronics cross sections.
    """

    from ._read_from_dict import read_from_xs_dict
    from ._read_from_file import read_from_xs_file

    def __init__(self):
        super().__init__()
        self.type = "XS"

        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.is_fissile: bool = False

        self._sigma_t: ndarray = []
        self._sigma_a: ndarray = []
        self.sigma_r: ndarray = []
        self.sigma_f: ndarray = []
        self.sigma_s: ndarray = []
        self.transfer_matrix: ndarray = []
        self.D: ndarray = []

        self.nu: ndarray = []
        self.nu_prompt: ndarray = []
        self.nu_delayed: ndarray = []

        self.chi: ndarray = []
        self.chi_prompt: ndarray = []
        self.chi_delayed: ndarray = []

        self.velocity: ndarray = []

        self.precursor_lambda: ndarray = []
        self.precursor_yield: ndarray = []

        self.sigma_t_function: XSFunc = None
        self.sigma_a_function: XSFunc = None

    def sigma_t(self, g: int, t: float = 0.0) -> float:
        if self.sigma_t_function is not None and \
                self.sigma_a_function is not None:
            raise AssertionError(
                "Only one transient cross section function can "
                "be provided.")

        if self.sigma_t_function is None and \
                self.sigma_a_function is None:
            return self._sigma_t[g]

        elif self.sigma_t_function is not None and \
                self.sigma_a_function is None:
            return self.sigma_t_function(g, t, self._sigma_t)

        elif self.sigma_a_function is not None and \
                self.sigma_t_function is None:
            val = self.sigma_a_function(g, t, self._sigma_a)
            return val + self.sigma_s[g]

    @property
    def nu_sigma_f(self) -> ndarray:
        """Get total nu times the fission cross sections.

        Returns
        -------
        ndarray (n_groups,)
        """
        return self.nu * self.sigma_f

    @property
    def nu_prompt_sigma_f(self) -> ndarray:
        """Get prompt nu times the fission cross sections.

        Returns
        -------
        ndarray (n_groups,)
        """
        return self.nu_prompt * self.sigma_f

    @property
    def nu_delayed_sigma_f(self) -> ndarray:
        """Get delayed nu times the fission cross sections.

        Returns
        -------
        ndarray (n_groups,)
        """
        return self.nu_delayed * self.sigma_f

    def _validate_xs(self) -> []:
        """Validate the parsed cross sections.
        """
        # Ensure sigma_t or sigma_a was provided
        has_sig_t = np.sum(self._sigma_t) > 0.0
        has_sig_a = np.sum(self._sigma_a) > 0.0
        if not has_sig_t and not has_sig_a:
            raise AssertionError(
                "Either the total or absorption cross sections "
                "must be provided.")

        # Compute sigma_s from transfer matrix
        self.sigma_s = np.sum(self.transfer_matrix, axis=1)

        # Enfore sigma_t = sigma_a + sigma_s
        if has_sig_a:
            self._sigma_t = self._sigma_a + self.sigma_s
        else:
            self._sigma_a = self._sigma_t - self.sigma_s

        # Compute diffusion coefficient, if not provided
        if np.sum(self.D) == 0.0:
            self.D = (3.0 * self._sigma_t) ** (-1.0)

        # Compute removal cross sections
        self.sigma_r = self._sigma_t - np.diag(self.transfer_matrix)

        # If not fissile with precursors
        if not self.is_fissile and self.n_precursors > 0:
            self.precursor_lambda = []
            self.precursor_yield = []
            self.chi_delayed = []
            self.n_precursors = 0

        # Check fissile properties
        if self.is_fissile:

            # Determine present nu terms
            has_nu = sum(self.nu) > 0.0
            has_nu_p = sum(self.nu_prompt) > 0.0
            has_nu_d = sum(self.nu_delayed) > 0.0

            # Determine present chi terms
            has_chi = sum(self.chi) > 0.0
            has_chi_p = sum(self.chi_prompt) > 0.0
            has_chi_d = np.sum(self.chi_delayed) > 0.0

            # Check precursor terms
            if self.n_precursors > 0:
                for j in range(self.n_precursors):
                    if self.precursor_lambda[j] == 0.0:
                        raise ValueError(
                            f"Precursor family {j} decay constant "
                            f"must be non-zero.")
                    if self.precursor_yield[j] == 0.0:
                        raise ValueError(
                            f"Precursor family {j} yield must be non-zero.")

            # Normalizaiton
            if has_chi:
                self.chi /= sum(self.chi)
            if has_chi_p:
                self.chi_prompt /= sum(self.chi_prompt)
            if has_chi_d:
                self.chi_delayed /= np.sum(self.chi_delayed, axis=0)
            if self.n_precursors > 0:
                self.precursor_yield /= sum(self.precursor_yield)

            # Check input compatibility
            if (has_nu_p and not has_chi_p) or \
                    (not has_nu_p and has_chi_p):
                raise AssertionError(
                    "If prompt nu or chi are provided, the other must be.")
            if (has_nu_d and not has_chi_d) or \
                    (not has_nu_d and has_chi_d):
                raise AssertionError(
                    "If delayed nu or chi are provided, the other must be.")
            if (has_nu and not has_chi) or \
                    (not has_nu and has_chi):
                raise AssertionError(
                    "If total nu or chi are provided, the other must be.")
            if (has_nu_p and not has_nu_d) or \
                    (not has_nu_p and has_nu_d):
                raise AssertionError(
                    "If prompt quantities are provided, delayed must be.")

            # Compute total from prompt and delayed
            if has_nu_p and has_nu_d:
                self.nu = self.nu_prompt + self.nu_delayed

            if has_chi_p and has_chi_d:
                beta = self.nu_delayed / self.nu
                self.chi = (1.0 - beta) * self.chi_prompt
                for j in range(self.n_precursors):
                    self.chi += beta * self.precursor_yield[j] *\
                                self.chi_delayed[:, j]

    def initialize_groupwise_data(self) -> []:
        """Initialize the group-wise only data.
        """
        # General cross sections
        self._sigma_t = np.zeros(self.n_groups)
        self._sigma_a = np.zeros(self.n_groups)
        self.sigma_s = np.zeros(self.n_groups)
        self.sigma_r = np.zeros(self.n_groups)
        self.sigma_f = np.zeros(self.n_groups)
        self.D = np.zeros(self.n_groups)

        # Transfer matrix
        self.transfer_matrix = np.zeros((self.n_groups,) * 2)

        # Fission neutron production data
        self.nu = np.zeros(self.n_groups)
        self.nu_prompt = np.zeros(self.n_groups)
        self.nu_delayed = np.zeros(self.n_groups)

        # Fission spectrum data
        self.chi = np.zeros(self.n_groups)
        self.chi_prompt = np.zeros(self.n_groups)

        # Groupwise velocities
        self.velocity = np.zeros(self.n_groups)

    def initialize_precursor_data(self) -> []:
        """Initialize the delayed neutron data.
        """
        G, J = self.n_groups, self.n_precursors
        self.precursor_lambda = np.zeros(J)
        self.precursor_yield = np.zeros(J)
        self.chi_delayed = np.zeros((G, J))

