from typing import List

from pyPDEs.utilities.boundaries import Boundary


class IncidentIsotropicFlux(Boundary):
    """Incident isotropic flux boundary condition.
    """

    def __init__(self, phi: List[float]) -> None:
        """Constructor.

        Parameters
        ----------
        phi : List[float]
            The group-wise incident isotropic flux.
        """
        super().__init__()
        self.type: str = "ISOTROPIC"
        self.values: List[float] = phi


class IncidentAngularFlux(Boundary):
    """Incident angular flux boundary condition.
    """

    def __init__(self, psi: List[List[float]]) -> None:
        """Constructor.

        Parameters
        ----------
        psi : List[List[float]
            The angle-wise, group-wise incident angular flux.
            The first index is for angle and the second for group.
        """
        self.type = "ANGULAR"
        self.values: List[List[float]] = psi


class VacuumBoundary(Boundary):
    """Vacuum boundary condition.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type: str = "VACUUM"


class ReflectiveBoundary(Boundary):
    """Reflective boundary condition.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type: str = "REFLECTIVE"


__all__ = ["IncidentIsotropicFlux", "IncidentAngularFlux",
           "VacuumBoundary", "ReflectiveBoundary"]
