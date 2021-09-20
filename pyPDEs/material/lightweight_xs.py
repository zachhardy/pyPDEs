import numpy as np

from .cross_sections import CrossSections


class LightWeightCrossSections:

    def __init__(self, xs: CrossSections) -> None:
        self._xs = xs
        self.sigma_t = xs.sigma_t
        self.D = xs.D

    def update(self, t: float) -> None:
        self.D = self._xs.D
        if self._xs.sigma_a_function is not None:
            f_sig_a = self._xs.sigma_a_function
            for g in range(self._xs.n_groups):
                sig_a = f_sig_a(g, t, self._xs.sigma_a)
                self.sigma_t[g] = sig_a + self._xs.sigma_s[g]
