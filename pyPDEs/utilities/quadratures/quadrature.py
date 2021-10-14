import numpy as np
from numpy import ndarray

from typing import List, Tuple

from .. import Vector


class Quadrature:
    """Base class for quadratures.
    """

    def __init__(self, order: int = 2) -> None:
        """Quadrature constructor.

        Parameters
        ----------
        order : int, default 2
            The maximum polynomial order the quadrature set
            can integrate exactly.
        """
        self.order: int = order
        self.qpoints: List[Vector] = None
        self.weights: List[float] = None
        self._domain: Tuple[float, float] = None  # only for 1D

    @property
    def n_qpoints(self) -> int:
        """Get the number of quadrature points.

        Returns
        -------
        int
        """
        return len(self.qpoints)

    def get_domain(self) -> Tuple[float, float]:
        return self._domain

    def set_domain(self, domain: Tuple[float, float]) -> None:
        new_domain = domain
        old_domain = self.get_domain()

        h_new = new_domain[1] - new_domain[0]
        h_old = old_domain[1] - old_domain[0]

        assert h_new > 0.0, "Invalid quadrature range."
        assert self.n_qpoints > 0, "Quadrature not initialized."

        for i in range(self.n_qpoints):
            f = h_new/h_old
            self.qpoints[i].z = \
                new_domain[0] + f*(self.qpoints[i].z - old_domain[0])
            self.weights *= f
        self._domain = domain


