import numpy as np

from .cross_sections import CrossSections


class LightWeightCrossSections:
    """
    A utility for storing cross-sections which are modified during
    a simulation. This is largely used when feedback or time-dependent
    cross-sections are utilized.
    """

    def __init__(self, xs: CrossSections) -> None:
        self._xs: CrossSections = xs
        self.sigma_t: np.ndarray = xs.sigma_t.copy()

    def update(self, args: list[float]) -> None:
        """
        Update the total cross-section by querying cross-section
        functions in the underlying CrossSections object with the
        specified arguments.

        Parameters
        ----------
        args : list[float]
            The arguments for the update functions.
        """
        if self._xs.sigma_a_function is not None:
            f = self._xs.sigma_a_function
            for g in range(self._xs.n_groups):
                sig_a = f(g, args, self._xs.sigma_a[g])
                self.sigma_t[g] = sig_a + self._xs.sigma_s[g]
