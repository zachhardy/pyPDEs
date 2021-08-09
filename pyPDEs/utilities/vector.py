from typing import Union


class Point:
    """
    Class for a 3-dimensional point.
    """
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0) -> None:
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __str__(self) -> str:
        return f"Point ({self.x}, {self.y}, {self.z})"

    def __add__(self, other: 'Point') -> 'Point':
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only addition by {cls_name} is allowed.")
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Point(x, y, z)

    def __iadd__(self, other: 'Point') -> None:
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only inplace addition by {cls_name} is allowed.")
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def __sub__(self, other: 'Point') -> 'Point':
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only subtraction by {cls_name} is allowed.")
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Point(x, y, z)

    def __isub__(self, other: 'Point') -> None:
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only inplace subtraction by {cls_name} is allowed.")
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def __mul__(self, other: float) -> 'Point':
        if not isinstance(other, float):
            raise TypeError(
                f"Only multiplication by float is allowed.")
        x = other * self.x
        y = other * self.y
        z = other * self.z
        return Point(x, y, z)

    def __rmul__(self, other: float) -> 'Point':
        if not isinstance(other, float):
            raise TypeError(
                f"Only right multiplication by float is allowed.")
        return self * other

    def __imul__(self, other: float) -> None:
        if not isinstance(other, float):
            raise TypeError(
                f"Only inplace multiplication by float is allowed.")
        self.x *= other
        self.y *= other
        self.z *= other

    def __truediv__(self, other: float) -> 'Point':
        if not isinstance(other, float):
            raise TypeError(
                f"Only division by float is allowed.")
        x = self.x / other
        y = self.y / other
        z = self.z / other
        return Point(x, y, z)

    def __itruediv__(self, other: float) -> None:
        if not isinstance(other, float):
            raise TypeError(
                f"Only inplace division by float is allowed.")
        self.x /= other
        self.y /= other
        self.z /= other

    def __abs__(self) -> 'Point':
        return Point(abs(self.x), abs(self.y), abs(self.z))

    def __neg__(self) -> 'Point':
        return Point(-self.x, -self.y, -self.z)

    def __eq__(self, other: 'Point') -> bool:
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Comparison can only be made to {cls_name}.")
        x_eq = self.x == other.x
        y_eq = self.y == other.y
        z_eq = self.z == other.z
        return x_eq and y_eq and z_eq

    def __ne__(self, other: 'Point'):
        return not self == other
