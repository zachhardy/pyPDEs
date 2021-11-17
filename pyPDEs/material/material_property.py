"""Simple material properties."""

import numpy as np
from numpy import ndarray

__all__ = ['ScalarProperty', 'IsotropicMultiGroupSource']


class MaterialProperty:
    """
    Generic material property.
    """
    def __init__(self) -> None:
        self.type: str = None


class ScalarProperty(MaterialProperty):
    """
    Scalar valued material property.

    Parameters
    ----------
    value : float, default 1.0
        The value of the scalar property.
    """
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.type = 'scalar'
        self.value: float = value


class IsotropicMultiGroupSource(MaterialProperty):
    """
    Multi-group source for neutronics.

    Parameters
    ----------
    values : ndarray
        The group-wise values of the source..
    """
    def __init__(self, values: ndarray) -> None:
        super().__init__()
        self.type = 'isotropic_source'
        self.values: ndarray = np.array(values)
        self.n_groups: int = len(self.values)
