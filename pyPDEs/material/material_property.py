"""Simple material properties."""

import numpy as np
from numpy import ndarray


class MaterialProperty:
    """Generic material property.

    Attributes
    ----------
    type : str
        The material property type. This is set in the
        constructor of derived classes.
    """
    def __init__(self) -> None:
        self.type: str = None


class ScalarProperty(MaterialProperty):
    """Scalar valued material property.

    Attributes
    ----------
    type : str
        The material property type. This is set in the
        constructor of derived classes.
    value : float
        The value of the scalar property.
    """
    def __init__(self, value: float = 1.0) -> None:
        """Class constructor.

        Parameters
        ----------
        value : float, default 1.0
        """
        super().__init__()
        self.type = "SCALAR"
        self.value: float = value


class MultiGroupSource(MaterialProperty):
    """Multi-group source for neutronics.

    Attributes
    ----------
    type : str
        The material property type. This is set in the
        constructor of derived classes.
    values : ndarray
        The group-wise values of the source.
    n_groups : int
        The number of energy groups.

    """
    def __init__(self, values: ndarray) -> None:
        super().__init__()
        self.type = "MULTIGROUP_SOURCE"
        self.values: ndarray = np.array(values)
        self.n_groups: int = len(self.values)

__all__ = ["ScalarProperty", "MultiGroupSource"]
