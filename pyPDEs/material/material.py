import numpy as np
from numpy import ndarray
from typing import List, NewType

Properties = NewType('Properties', List['MaterialProperty'])


class Material:
    """
    Base class for a material. A material is defined as collection
    of material properties.

    Attributes
    ----------
    properties : List[MaterialProperty]
    name : str, default "Generic Material"
    """
    def __init__(self) -> None:
        self.properties: Properties = []
        self.name: str = "Generic Material"

    def add_properties(self, properties: Properties) -> None:
        """
        Add material properties to this material.

        Parameters
        ----------
        properties : List[MaterialProperty]
        """
        if not isinstance(properties, list):
            properties = [properties]
        self.properties.extend(properties)


class MaterialProperty:
    """
    Base class for a material property.

    Attributes
    ----------
    type : str
    name : str

    Parameters
    ----------
    property_type : str
    """
    def __init__(self, property_type: str) -> None:
        self.type: str = property_type
        self.name: str = None


class ScalarProperty(MaterialProperty):
    """
    Class for a scalar property.

    Attributes
    ----------
    value : float

    Parameters
    ----------
    value : float, default 1.0
        The scalar value to set.
    """
    def __init__(self, value: float = 1.0) -> None:
        super().__init__("SCALAR")
        self.value: float = value


class MultiGroupSource(MaterialProperty):
    """
    Class for a multi-group source.

    Attributes
    ----------
    values : ndarray

    Parameters
    ----------
    values : ndarray
        The multi-group source values to set.
    """
    def __init__(self, values: ndarray) -> None:
        super().__init__("MULTIGROUP_SOURCE")
        self.values: ndarray = np.array(values)

    @property
    def num_components(self) -> int:
        """
        Get the number of components of the multi-group source.

        Returns
        -------
        int
        """
        return len(self.values)

