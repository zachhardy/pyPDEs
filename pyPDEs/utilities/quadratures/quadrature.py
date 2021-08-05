from typing import List, Tuple

import numpy as np
from numpy import ndarray


class Quadrature:
    """
    Base class for quadratures.
    """
    def __init__(self, order: int = 2) -> None:
        self.order: int = order
        self.qpoints: ndarray = None
        self.weights: ndarray = None
        self.domain: Tuple[float] = None
        self.width: float = 0.0

    @property
    def n_qpoints(self) -> int:
        return len(self.qpoints)
