import numpy as np

from typing import Union


class CartesianVector:
    """
    Implementation of a general three-vector.

    Attributes
    ----------
    _xyz : numpy.ndarray
    """

    def __init__(
            self,
            x: float = 0.0,
            y: flaot = 0.0,
            z: float = 0.0
    ) -> None:
        """
        Construct a Cartesian vector from an x, y, and z coordinate.

        Parameters
        ----------
        x : float, The x-coordinate.
        y : float, The y-coordinate.
        z : float, The z-coordinate.
        """

        self._xyz = np.array([x, y, z])

    def __iadd__(self, other: CartesianVector) -> CartesianVector:
        """
        Add another Cartesian vector to this one.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        CartesianVector
        """
        if not isinstance(other, type(self)):
            msg = f"Cannot add {type(self)} to a Cartesian vector."
            raise TypeError(msg)
        self._xyz += other._xyz
        return self

    def __add__(self, other: CartesianVector) -> CartesianVector:
        """
        Add two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        CartesianVector
        """
        if not isinstance(other, type(self)):
            msg = f"Cannot add {type(self)} to a Cartesian vector."
            raise TypeError(msg)
        xyz = self._xyz + other._xyz
        return CartesianVector(*xyz.tolist())

    def __isub__(self, other: CartesianVector) -> CartesianVector:
        """
        Subtract another Cartesian vector from this one.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        CartesianVector
        """
        if not isinstance(other, type(self)):
            msg = f"Cannot subtract {type(self)} from a Cartesian vector."
            raise TypeError(msg)
        self._xyz -= other._xyz
        return self

    def __sub__(self, other: CartesianVector) -> CartesianVector:
        """
        Return the difference of two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        CartesianVector
        """
        if not isinstance(other, type(self)):
            msg = f"Cannot subtract {type(self)} from a Cartesian vector."
            raise TypeError(msg)
        xyz = self._xyz - other._xyz
        return CartesianVector(*xyz.tolist())

    def __imul__(self, factor: Union[int, float]) -> CartesianVector:
        """
        Multiply this Cartesian vector by a scalar.

        Parameters
        ----------
        factor : float

        Returns
        -------
        CartesianVector
        """
        if isinstance(factor, int):
            factor = float(factor)
        if not isinstance(factor, float):
            msg = f"Cannot multiply a Cartesian vector by a {type(factor)}."
            raise TypeError(msg)
        self._xyz *= factor
        return self

    def __mul__(self, factor: Union[int, float]) -> CartesianVector:
        """
        Return a Cartesian vector multiplied by a scalar.

        Parameters
        ----------
        factor : float

        Returns
        -------
        CartesianVector
        """
        if isinstance(factor, int):
            factor = float(factor)
        if not isinstance(factor, float):
            msg = f"Cannot multiply a Cartesian vector by a {type(factor)}."
            raise TypeError(msg)
        xyz = factor * self._xyz
        return CartesianVector(*xyz.tolist())

    def __rmul__(self, factor: Union[int, float]) -> CartesianVector:
        """
        Return a Cartesian vector left-multiplied by a scalar.

        Parameters
        ----------
        factor : float

        Returns
        -------
        CartesianVector
        """
        return self * factor

    def __itruediv__(self, factor: Union[int, float]) -> CartesianVector:
        """
        Divide the Cartesian vector by a scalar.

        Parameters
        ----------
        factor : float

        Returns
        -------
        CartesianVector
        """
        if isinstance(factor, int):
            factor = float(factor)
        if not isinstance(factor, float):
            msg = f"Cannot divide a Cartesian vector by a {type(factor)}."
            raise TypeError(msg)
        self._xyz /= factor
        return self

    def __truediv__(self, factor: Union[int, float]) -> CartesianVector:
        """
        Return a Cartesian vector divided by a scalar.

        Parameters
        ----------
        factor : float

        Returns
        -------
        CartesianVector
        """
        if isinstance(factor, int):
            factor = float(factor)
        if not isinstance(factor, float):
            msg = f"Cannot divide a Cartesian vector by a {type(factor)}."
            raise TypeError(msg)
        xyz = self._xyz / factor
        return CartesianVector(*xyz.tolist())

    def __abs__(self) -> CartesianVector:
        """
        Return the absolute value of the Cartesian vector.

        Returns
        -------
        CartesianVector
        """
        return CartesianVector(*self._xyz.tolist())

    def __neg__(self) -> CartesianVector:
        """
        Return the negative of the Cartesian vector.

        Returns
        -------
        CartesianVector
        """
        return CartesianVector(*(-self._xyz).tolist())

    def __setitem__(self, index: int, value: float) -> None:
        """
        Set an element of the Cartesian vector.

        Parameters
        ----------
        index : int
        value : float
        """
        if index > 2:
            msg = "Index must be either 0(x), 1(y), or 2(z)."
            raise ValueError(msg)
        self._xyz[index] = value

    def __getitem__(self, index: int) -> float:
        """
        Return an element of the Cartesian vector.

        Parameters
        ----------
        index : int

        Returns
        -------
        float
        """
        if index > 2:
            msg = "Index must be either 0(x), 1(y), or 2(z)."
            raise ValueError(msg)
        return self._xyz[index]

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"CartesianVector{str(self)}"

    def __eq__(self, other: CartesianVector) -> CartesianVector:
        """
        Test the equality of two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        bool
        """
        return all(self._xyz == other._xyz)

    def __ne__(self, other: CartesianVector) -> CartesianVector:
        """
        Test the inequality of two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        float
        """
        return self._xyz != other._xyz

    @property
    def x(self) -> float:
        """
        Return the x-coordinate.

        Returns
        -------
        float
        """
        return self._xyz[0]

    @x.setter
    def x(self, value: float) -> None:
        """
        Set the x-coordinate.

        Parameters
        ----------
        value : float
        """
        self._xyz[0] = value

    @property
    def y(self)-> float:
        """
        Return the y-coordinate.

        Returns
        -------
        float
        """
        return self._xyz[1]

    @y.setter
    def y(self, value: float) -> None:
        """
        Set the y-coordinate.

        Parameters
        ----------
        value : float
        """
        self._xyz[1] = value

    @property
    def z(self) -> float:
        """
        Return the z-coordinate

        Returns
        -------
        float
        """
        return self._xyz[2]

    @z.setter
    def z(self, value: float) -> None:
        """
        Set the z-coordinate.

        Parameters
        ----------
        value : float
        """
        self._xyz[2] = value

    def dot(self, other: CartesianVector) -> float:
        """
        Return the dot product between two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        float
        """
        if not isinstance(other, type(self)):
            msg = f"Cannot dot a Cartesian vector with {type(other)}."
            raise TypeError(msg)
        return np.dot(self._xyz, other._xyz)

    def cross(self, other: CartesianVector) -> CartesianVector:
        """
        Return the cross product between two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        CartesianVector
        """
        if not isinstance(other, type(self)):
            msg = f"Cannot cross a Cartesian vector with {type(other)}."
            raise TypeError(msg)
        x = self.y * other.z - other.y * self.z
        y = other.x * self.z - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return CartesianVector(x, y, z)

    def length(self) -> float:
        """
        Return the Euclidean distance to the origin.

        Returns
        -------
        float
        """
        return np.sqrt(self.dot(self))

    def direction(self) -> CartesianVector:
        """
        Return the direction of the Cartesian vector.

        Returns
        -------
        CartesianVector
        """
        length = self.length()
        return self / length if length > 0.0 else self

    def distance(self, other: CartesianVector) -> float:
        """
        Return the Euclidean distance between two Cartesian vectors.

        Parameters
        ----------
        other : CartesianVector

        Returns
        -------
        float
        """
        return (self - other).length()


def dot(p: CartesianVector, q: CartesianVector) -> float:
    """
    Return the dot product between two Cartesian vectors.

    Parameters
    ----------
    p : CartesianVector
    q : CartesianVector

    Returns
    -------
    double
    """
    return p.dot(q)


def cross(p: CartesianVector, q: CartesianVector) -> CartesianVector:
    """
    Return the cross product between two Cartesian vectors.

    Parameters
    ----------
    p : CartesianVector
    q : CartesianVector

    Returns
    -------
    CartesianVector
    """
    return p.cross(q)


def distance(p: CartesianVector, q: CartesianVector) -> float:
    """
    Return the distance between two Cartesian vectors.

    Parameters
    ----------
    p : CartesianVector
    q : CartesianVector

    Returns
    -------
    CartesianVector
    """
    return p.distance(q)


def direction(p: CartesianVector) -> CartesianVector:
    """
    Return the direction of a Cartesian vector.

    Parameters
    ----------
    p : CartesianVector

    Returns
    -------
    CartesianVector
    """
    return p.direction()

