from typing import List

from pyPDEs.utilities.boundaries import Boundary

__all__ = ['IncidentHomogeneousBoundary', 'VacuumBoundary']


class IncidentHomogeneousBoundary(Boundary):
    """
    Incident homogeneous transport boundary condition.

    Parameters
    ----------
    values : List[float]
        The boundary value, or values.
    """
    def __init__(self, values: List[float]) -> None:
        super().__init__()
        self.type = 'incident_homogeneous'
        if isinstance(values, float):
            values = [values]
        self.values: List[float] = values


class VacuumBoundary(IncidentHomogeneousBoundary):
    """
    Vacuum transport boundary condition.

    Parameters
    ----------
    n_groups : int, default 1
    """
    def __init__(self, n_groups: int = 1) -> None:
        super().__init__([0.0] * n_groups)
        self.type = 'vacuum'
