from numpy import ndarray
from typing import List

from pyPDEs.utilities import Vector


__all__ = ['ReflectiveBoundary', 'VacuumBoundary',
           'HomogeneousBoudary', 'Boundary']


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

        self.angle_ready_flags: List[bool] = []

        values = [] if values is None else values
        if isinstance(values, float):
            values = [values]
        self.values: List[float] = values

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

    def set_angle_ready_status(self, angles: List[int]) -> None:
        return

    def check_angle_ready_status(self, angles: List[int]) -> bool:
        return True


class HomogeneousBoudary(Boundary):
    """
    Incident homogeneous transport boundary condition.

    Parameters
    ----------
    values : List[float], default None
        Group-wise homogeneous boundary values.
    """
    def __init__(self, values: List[float]) -> None:
        super().__init__('homogeneous', values)


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
        AngVec = List[List[List[float]]]

        super().__init__('reflective')
        self.normal: Vector = None
        self.boundary_psi: List[AngVec] = []
        self.reflected_angles: List[int] = []

    def boundary_psi_incoming(self, angle_num: int, cell_id: int,
                              face_num: int) -> float:
        """
        Get the incident boundary psi.

        Parameters
        ----------
        angle_num : int
            The angle index.
        cell_id : int
            The cell index.
        face_num : int
            The face index.

        Returns
        -------
        float
        """
        refl = self.reflected_angles[angle_num]
        return self.boundary_psi[refl][cell_id][face_num]

    def set_psi_outgoing(self, psi: float, cell_id: int, face_num: int,
                         angle_num: int, group_num: int) -> None:
        """
        Get the outgoing boundary psi.

        Parameters
        ----------
        psi: float
        angle_num : int
        cell_id : int
        face_num : int
        group_num : int
        """
        self.boundary_psi[angle_num][cell_id][face_num][group_num] = psi

    def set_angle_ready_status(self, angles: List[int]) -> None:
        for n in angles:
            self.angle_ready_flags[self.reflected_angles[n]] = True

    def check_angle_ready_status(self, angles: List[int]) -> bool:
        for n in angles:
            if len(self.boundary_psi[self.reflected_angles[n]]) > 0:
                if not self.angle_ready_flags[n]:
                    return False
        return True

    def reset_angle_ready_status(self) -> None:
        for flag in self.angle_ready_flags:
            flag = False

