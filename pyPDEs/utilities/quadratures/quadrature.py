import numpy as np
from numpy import ndarray

from typing import List, Tuple

from .. import Vector


class Quadrature:
    """
    Base class for quadratures.

    Parameters
    ----------
    order : int, default 2
        The maximum monomial order the quadrature set
        can integrate exactly.
    """
    def __init__(self, order: int = 2) -> None:
        self.order: int = order
        self.qpoints: List[Vector] = None
        self.weights: List[float] = None
        self.domain: Tuple[float] = None  # only for 1D

    @property
    def n_qpoints(self) -> int:
        return len(self.qpoints)
