import numpy as np

from typing import Union


class MaterialProperty:
    """
    Base class for a material property.
    """

    def __init__(self, property_type: str) -> None:
        """
        Parameters
        ----------
        property_type : str
        """
        self.type: str = property_type


class ScalarProperty(MaterialProperty):
    """
    A class for scalar valued material properties.
    """

    def __init__(self, value: float = 1.0) -> None:
        """
        Parameters
        ----------
        value : float
        """
        if not isinstance(value, float):
            msg = "The specified value must be a float."
            raise TypeError(msg)

        super().__init__("SCALAR")
        self.value: float = value


class IsotropicMultiGroupSource(MaterialProperty):
    """
    A class for isotropic multi-group neutronics sources.
    """

    def __init__(self, values: Union[list[float], np.ndarray]) -> None:
        """
        Parameters
        ----------
        values : numpy.ndarray or list[float]
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, list):
            msg = "The specified values must be a numpy ndarray or a list."
            raise TypeError(msg)

        super().__init__("ISOTROPIC_SOURCE")
        self.values: np.ndarray = np.array(values)
