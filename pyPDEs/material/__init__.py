import numpy as np
from numpy import ndarray
from typing import List, NewType

from .material_property import MaterialProperty
from .material_property import ScalarProperty, IsotropicMultiGroupSource
from .cross_sections import CrossSections
from .lightweight_xs import LightWeightCrossSections

Properties = List[MaterialProperty]


class Material:
    """Generic material.

    Attributes
    ----------
    properties : List[MaterialProperty]
        A list of properties that define this material.
    name : str
        A name for the material. This is currently unused.
    """
    def __init__(self, name: str = "Generic Material") -> None:
        self.properties: Properties = []
        self.name: str = name

    def add_properties(self, properties: Properties) -> None:
        """Add material properties to the material.

        Parameters
        ----------
        properties : List[MaterialProperty]
            A list of properties to add to this material.
        """
        if not isinstance(properties, list):
            properties = [properties]
        self.properties.extend(properties)


__all__ = ["Material", "ScalarProperty", "IsotropicMultiGroupSource",
           "CrossSections", "LightWeightCrossSections"]
