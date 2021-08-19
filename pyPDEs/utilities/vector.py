from typing import Union

import numpy as np


class Vector:
    """Class for a 3-dimensional vector.

    Attributes
    ----------
    x : float
        The x-coordinate.
    y : float
        The y-coordinate.
    z : float
        The z-coordinate.
    """
    def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
        """Constructor.

        Parameters
        ----------
        x : float
        y : float
        z : float
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __str__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"

    def __add__(self, other: "Vector") -> "Vector":
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only addition by {cls_name} is allowed.")
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return Vector(x, y, z)

    def __iadd__(self, other: "Vector") -> "Vector":
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only inplace addition by {cls_name} is allowed.")
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __sub__(self, other: "Vector") -> "Vector":
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only subtraction by {cls_name} is allowed.")
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return Vector(x, y, z)

    def __isub__(self, other: "Vector") -> "Vector":
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only inplace subtraction by {cls_name} is allowed.")
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z
        return self

    def __mul__(self, other: Union[float, int, "Vector"]) -> "Vector":
        if isinstance(other, (float, int)):
            x = other * self.x
            y = other * self.y
            z = other * self.z
            return Vector(x, y, z)
        elif isinstance(other, type(self)):
            x = other.x * self.x
            y = other.y * self.y
            z = other.z * self.z
        else:
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only multiplication by float, int, or "
                f"{cls_name} is allowed.")
        return Vector(x, y, z)

    def __rmul__(self, other: Union[float, int, "Vector"]) -> "Vector":
        if not isinstance(other, (float, int, type(self))):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only right multiplication by float, int, or "
                f"{cls_name} is allowed.")
        return self * other

    def __imul__(self, other: Union[float, int, "Vector"]) -> "Vector":
        if isinstance(other, (float, int)):
            self.x *= other
            self.y *= other
            self.z *= other
        elif isinstance(other, type(self)):
            self.x *= other.x
            self.y *= other.y
            self.z *= other.z
        else:
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only inplace multiplication by float or "
                f"{cls_name} is allowed.")
        return self

    def __truediv__(self, other: Union[float, int, "Vector"]) -> "Vector":
        if isinstance(other, (float, int)):
            x = self.x / other
            y = self.y / other
            z = self.z / other
        elif isinstance(other, type(self)):
            x = self.x / other.x
            y = self.y / other.y
            z = self.z / other.z
        else:
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only division by float or {cls_name} is allowed.")
        return Vector(x, y, z)

    def __itruediv__(self, other: Union[float, int, "Vector"]) -> "Vector":
        if isinstance(other, (float, int)):
            self.x /= other
            self.y /= other
            self.z /= other
        elif isinstance(other, type(self)):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
        else:
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Only inplace division by float or {cls_name} is allowed.")
        return self

    def __pow__(self, power: Union[float, int]) -> "Vector":
        if not isinstance(power, (float, int)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Exponentiation only allowed with float or int.")
        x = self.x ** power
        y = self.y ** power
        z = self.z ** power
        return Vector(x, y, z)

    def __ipow__(self, power: Union[float, int]) -> "Vector":
        if not isinstance(power, (float, int)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Inplace exponentiation only allowed with float or int.")
        self.x **= power
        self.y **= power
        self.z **= power
        return self

    def __abs__(self) -> "Vector":
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y, -self.z)

    def __eq__(self, other: "Vector") -> bool:
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Comparison can only be made to {cls_name}.")
        x_eq = self.x == other.x
        y_eq = self.y == other.y
        z_eq = self.z == other.z
        return x_eq and y_eq and z_eq

    def __ne__(self, other: "Vector"):
        return not self == other

    def dot(self, other: "Vector") -> float:
        """Compute the dot product of this with another Vector.

        Parameters
        ----------
        other : Vector

        Returns
        -------
        float
        """
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Dot products must be with {cls_name}.")
        dot = self.x * other.x
        dot += self.y * other.y
        dot += self.z * other.z
        return dot

    def cross(self, other: "Vector") -> "Vector":
        """Compute the cross product of this with another Vector.

        Parameters
        ----------
        other : Vector

        Returns
        -------
        Vector
        """
        if not isinstance(other, type(self)):
            cls_name = self.__class__.__name__
            raise TypeError(
                f"Cross products must be with {cls_name}.")
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)

    def norm(self) -> float:
        """Compute the norm of this.

        Returns
        -------
        float
        """
        return np.sqrt(self.dot(self))

    def normalize(self) -> "Vector":
        """Normalize this vector to unit length.

        Returns
        -------
        Vector
            This Vector, but normalized to unit length.
        """
        norm = self.norm()
        self /= norm
        return self


