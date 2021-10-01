import numpy as np
from typing import List

from .cross_sections import CrossSections


class LightWeightCrossSections:

    def __init__(self, xs: CrossSections) -> None:
        self._xs = xs
        self.sigma_t = np.copy(xs.sigma_t)
        self.D = np.copy(xs.D)
        self.B_sq = np.copy(xs.B_sq)

    def update(self, x: List[float]) -> None:
        if self._xs.sigma_a_function is not None:
            f_sig_a = self._xs.sigma_a_function
            for g in range(self._xs.n_groups):
                sig_a = f_sig_a(g, x, self._xs.sigma_a[g])
                self.sigma_t[g] = sig_a + self._xs.sigma_s[g]
