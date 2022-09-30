from .material_property import MaterialProperty


class Material:
    """
    A class describing a material.

    Materials are made up of properties which are consumed by
    physics solvers.
    """

    def __init__(self, name: str = "Generic Material") -> None:
        """
        Parameters
        ----------
        name : str
        """
        self.name: str = name
        self.properties: list[MaterialProperty] = []
