from numpy import ndarray
from typing import List

from pyPDEs.utilities import Vector


__all__ = ['ReflectiveBoundary', 'VacuumBoundary',
           'IncidentHomogeneousBoundary']


class Boundary:
    """
    Base class for a transport boundary.

    Parameters
    ----------
    bndry_type : str
        A string identifier for the boundary type.
    values : List[float], default None
        Group-wise homogeneous boundary values.
    """
    def __init__(self, bndry_type: str,
                 values: List[float] = None) -> None:
        self.type: str = bndry_type

        self.values: List[float] = []
        if values is not None:
            if isinstance(values, float):
                values = [values]
            for value in values:
                self.values.append(value)

    def boundary_psi_incoming(
            self, angle: int, group: int, cell: int,
            face: int, node: int) -> float:
        raise AssertionError(
            'No boundary psi exists for this boundary type.')

    def boundary_psi_outgoing(
            self, angle: int, group: int, cell: int,
            face: int, node: int) -> float:
        raise AssertionError(
            'No boundary psi exists for this boundary type.')


class IncidentHomogeneousBoundary(Boundary):
    """
    Incident homogeneous transport boundary condition.

    Parameters
    ----------
    values : List[float], default None
        Group-wise homogeneous boundary values.
    """
    def __init__(self, values: List[float]) -> None:
        super().__init__('incident_homogeneous', values)


class VacuumBoundary(Boundary):
    """
    Vacuum transport boundary condition.
    """
    def __init__(self) -> None:
        super().__init__('vacuum')


class ReflectiveBoundary(Boundary):
    """
    Reflective transport boundary condition.
    """
    def __init__(self) -> None:
        AngVec = List[List[List[List[float]]]]

        self.normal: Vector = None
        self.boundary_psi: List[AngVec] = []
        self.reflected_angle: List[int] = []

    def boundary_psi_incoming(
            self, angle: int, group: int, cell: int,
            face: int, node: int) -> float:
        """
        Get the incident boundary psi.

        Parameters
        ----------
        angle : int
            The angle index.
        group : int
            The group index.
        cell : int
            The cell index.
        face : int
            The face index.
        node : int
            The node index.

        Returns
        -------
        float
        """
        psi_n = self.boundary_psi[self.reflected_angle[angle]]
        return psi_n[group][cell][face][node]

    def boundary_psi_outgoing(
            self, angle: int, group: int, cell: int,
            face: int, node: int) -> float:
        """
        Get the outgoing boundary psi.

        Parameters
        ----------
        angle : int
            The angle index.
        group : int
            The group index.
        cell : int
            The cell index.
        face : int
            The face index.
        node : int
            The node index.

        Returns
        -------
        float
        """
        return self.boundary_psi[angle][group][cell][face][node]



