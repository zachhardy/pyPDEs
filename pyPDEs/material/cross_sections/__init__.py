import numpy as np

from numpy import ndarray
from typing import Callable

from ..material_property import MaterialProperty

XSFunc = Callable[[int, float], float]


class CrossSections(MaterialProperty):
    """
    Neutronics cross sections.
    """

    from ._read_from_dict import read_from_xs_dict
    from ._read_from_file import read_from_xs_file

    def __init__(self):
        super().__init__()
        self.type = 'xs'

        self.density: float = 1.0
        self.k_eff: float = 1.0

        self.n_groups: int = 0
        self.n_precursors: int = 0
        self.scattering_order: int = 0
        self.is_fissile: bool = False

        self._sigma_t: ndarray = []
        self._sigma_a: ndarray = []
        self._sigma_r: ndarray = []
        self._sigma_f: ndarray = []
        self._sigma_s: ndarray = []
        self._transfer_matrix: ndarray = []
        self._D: ndarray = []
        self.B_sq: ndarray = []

        self.nu: ndarray = []
        self.nu_prompt: ndarray = []
        self.nu_delayed: ndarray = []

        self.chi: ndarray = []
        self.chi_prompt: ndarray = []
        self.chi_delayed: ndarray = []

        self.velocity: ndarray = []

        self.precursor_lambda: ndarray = []
        self.precursor_yield: ndarray = []

        self.sigma_a_function: XSFunc = None

    @property
    def sigma_t(self) -> ndarray:
        return self.density * self._sigma_t

    @property
    def sigma_a(self) -> ndarray:
        return self.density * self._sigma_a

    @property
    def sigma_r(self) -> ndarray:
        return self.density * self._sigma_r

    @property
    def sigma_f(self) -> ndarray:
        return self.density * self._sigma_f / self.k_eff

    @property
    def sigma_s(self) -> ndarray:
        return self.density * self._sigma_s

    @property
    def transfer_matrix(self) -> ndarray:
        return self.density * self._transfer_matrix

    @property
    def D(self) -> ndarray:
        if sum(self._D) == 0.0:
            return 1.0 / (3.0 * self.sigma_t)
        else:
            return self._D

    @property
    def nu_sigma_f(self) -> ndarray:
        return self.nu * self.sigma_f

    @property
    def nu_prompt_sigma_f(self) -> ndarray:
        return self.nu_prompt * self.sigma_f

    @property
    def nu_delayed_sigma_f(self) -> ndarray:
        return self.nu_delayed * self.sigma_f

    def _validate_xs(self) -> None:
        """
        Validate the parsed cross sections.
        """
        # Ensure sigma_t or sigma_a was provided
        has_sig_t = np.sum(self.sigma_t) > 0.0
        has_sig_a = np.sum(self.sigma_a) > 0.0
        if not has_sig_t and not has_sig_a:
            raise AssertionError(
                'Either the total or absorption cross sections '
                'must be provided.')

        # Compute sigma_s from transfer matrix
        self._sigma_s = np.sum(self._transfer_matrix[0], axis=1)

        # Enfore sigma_t = sigma_a + sigma_s
        if has_sig_a:
            self._sigma_t = self._sigma_a + self._sigma_s
        else:
            self._sigma_a = self._sigma_t - self._sigma_s

        # Compute removal cross sections
        self._sigma_r = self._sigma_t - np.diag(self._transfer_matrix[0])

        # Set fissile
        self.is_fissile = sum(self._sigma_f) > 0.0

        # If not fissile with precursors
        if not self.is_fissile and self.n_precursors > 0:
            self.n_precursors = 0
            self.precursor_lambda = np.zeros(0)
            self.precursor_yield = np.zeros(0)
            self.chi_delayed = np.zeros((self.n_groups, 0))

        # Check fissile properties
        if self.is_fissile:

            # Set chi for one group problems
            if self.n_groups == 1:
                self.chi = np.ones(1)
                if self.n_precursors > 0:
                    self.chi_prompt = np.ones(1)
                    self.chi_delayed = np.ones((1, self.n_precursors))

            # Check whether total quantities
            has_nu = sum(self.nu) > 0.0
            has_chi = sum(self.chi) > 0.0
            has_total = has_nu and has_chi
            if has_total:
                self.chi /= sum(self.chi)

            # Check prompt/delayed quantities
            has_nu_prompt = sum(self.nu_prompt) > 0.0
            has_chi_prompt = sum(self.chi_prompt) > 0.0
            has_prompt = has_nu_prompt and has_chi_prompt
            if has_prompt:
                self.chi_prompt /= sum(self.chi_prompt)

            if self.n_precursors > 0 and not has_prompt:
                raise AssertionError(
                    'Prompt quantities must be provided when precursors '
                    'are present in the cross section set.')

            has_nu_delayed = sum(self.nu_delayed) > 0.0
            has_chi_delayed = len(self.chi_delayed) > 0
            if has_chi_delayed:
                chi_dj_sums = np.sum(self.chi_delayed, axis=0)
                nonzero_chi_d = all([s > 0.0 for s in chi_dj_sums])
                if not nonzero_chi_d:
                    raise ValueError('All delayed chi spectra must have '
                                     'non-zero components')
                self.chi_delayed /= chi_dj_sums

            has_delayed = has_nu_delayed and has_chi_delayed
            if self.n_precursors > 0 and not has_delayed:
                raise AssertionError(
                    'Delayed quantities must be provided when precursors '
                    'are present in the cross section set.')

            # Check precursor properties
            if self.n_precursors > 0:
                lambda_ = self.precursor_lambda
                has_lambda = all([val > 0.0 for val in lambda_])
                has_gamma = sum(self.precursor_yield) > 0.0
                if has_gamma:
                    self.precursor_yield /= sum(self.precursor_yield)

                if not has_lambda:
                    raise AssertionError(
                        'All precursor decay constants must be strictly '
                        'greater than zero.')
                if not has_gamma:
                    raise AssertionError(
                        'The precursor yields must be sum to a positive '
                        'non-zero quantity.')

            # Compute total from prompt and delayed
            if has_prompt and has_delayed:
                self.nu = self.nu_prompt + self.nu_delayed

                beta = self.nu_delayed / self.nu
                self.chi = (1.0 - beta) * self.chi_prompt
                for j in range(self.n_precursors):
                    self.chi += beta * self.precursor_yield[j] * \
                                self.chi_delayed[:, j]

    def initialize_groupwise_data(self) -> None:
        """
        Initialize the group-wise only data.
        """
        # General cross sections
        self._sigma_t = np.zeros(self.n_groups)
        self._sigma_a = np.zeros(self.n_groups)
        self._sigma_s = np.zeros(self.n_groups)
        self._sigma_r = np.zeros(self.n_groups)
        self._sigma_f = np.zeros(self.n_groups)
        self._D = np.zeros(self.n_groups)
        self.B_sq = np.zeros(self.n_groups)

        # Fission neutron production data
        self.nu = np.zeros(self.n_groups)
        self.nu_prompt = np.zeros(self.n_groups)
        self.nu_delayed = np.zeros(self.n_groups)

        # Fission spectrum data
        self.chi = np.zeros(self.n_groups)
        self.chi_prompt = np.zeros(self.n_groups)

        # Groupwise velocities
        self.velocity = np.zeros(self.n_groups)

    def initialize_precursor_data(self) -> None:
        """
        Initialize the delayed neutron data.
        """
        G, J = self.n_groups, self.n_precursors
        self.precursor_lambda = np.zeros(J)
        self.precursor_yield = np.zeros(J)
        self.chi_delayed = np.zeros((G, J))

