import numpy as np
from typing import Union, Type, Iterator

Number = Union[int, float]
T = Type[Union[int, float]]


class CartesianVector:
    """
    Implementation of a general Cartesian vector.

    Based on the inputs, this can be 1D, 2D, or 3D.
    Additionally, if desired, ``dtype`` can be specified
    to fix the data-type of the underlying data.
    """

    def __init__(
            self,
            x: Number = 0.0,
            y: Number = None,
            z: Number = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        x : int or float, default 0.0
        y : int or float, default None
        z : int or float, default None.
        """
        # defaults
        dtype = type(x)
        y = dtype(0.0) if not y else y
        z = dtype(0.0) if not z else z

        # checks
        if not isinstance(x, (int, float)):
            raise TypeError("Invalid x-coordinate type.")
        if not isinstance(y, (int, float)):
            raise TypeError("Invalid y-coordinate type.")
        if not isinstance(z, (int, float)):
            raise TypeError("Invalid z-coordinate type.")

        # define the array
        self._xyz = np.array([x, y, z])

    def __len__(self) -> int:
        """Return the length, or dimension, of the vector."""
        return len(self._xyz)

    def __iter__(self) -> Iterator[Number]:
        """Return an iterator to the underlying numpy ndarray."""
        return iter(self._xyz)

    def __str__(self) -> str:
        """Return the vector as a string."""
        return str(self._xyz)

    def __repr__(self) -> str:
        """Return a string representation of the vector."""
        s = f"{self._xyz[0]}"
        if len(self._xyz) > 1:
            s += f", {self._xyz[1]}"
            if len(self._xyz) > 2:
                s += f", {self._xyz[2]}"
        return f"CartesianVector({s})"

    def __hash__(self) -> int:
        """Return a unique hash for the vector."""
        return hash(self._xyz.tobytes())

    def __eq__(self, other: 'CartesianVector') -> bool:
        """Return whether two vectors are equivalent."""
        return all(self._xyz == other._xyz)

    def __ne__(self, other: 'CartesianVector') -> bool:
        """Return whether two vectors are different."""
        return any(self._xyz != other._xyz)

    def __getitem__(self, item: int) -> Number:
        """Return an element from the vector."""
        if item > 2:
            raise IndexError(f"Out of range error.")
        return self._xyz[item]

    def __setitem__(self, key: int, value: Number) -> None:
        """Set an element of the vector."""
        if key > 2:
            raise IndexError(f"Out of range error.")
        self._xyz[key] = value

    def __abs__(self) -> 'CartesianVector':
        """Return the absolute value of the vector."""
        return CartesianVector(*np.abs(self._xyz))

    def __neg__(self) -> 'CartesianVector':
        """Negate the vector."""
        return CartesianVector(*(-self._xyz))

    def __add__(self, other: 'CartesianVector') -> 'CartesianVector':
        """Return the sum of two vectors."""
        if not isinstance(other, CartesianVector):
            raise TypeError(f"Only addition by {type(self)} is permitted.")
        xyz = self._xyz + other._xyz
        return CartesianVector(*xyz)

    def __sub__(self, other: 'CartesianVector') -> 'CartesianVector':
        """Return the difference between two vectors."""
        if not isinstance(other, CartesianVector):
            raise TypeError(f"Only subtraction by {type(self)} is permitted.")
        xyz = self._xyz - other._xyz
        return CartesianVector(*xyz)

    def __mul__(self, factor: Number) -> 'CartesianVector':
        """Return the vector multiplied by a scalar."""
        if not isinstance(factor, (int, float)):
            raise TypeError("Only scalar multiplication is permitted.")
        xyz = factor * self._xyz
        return CartesianVector(*xyz)

    def __truediv__(self, factor: Number) -> 'CartesianVector':
        """Return the vector divided by a scalar."""
        if not isinstance(factor, (int, float)):
            raise TypeError("Only scalar division is permitted.")
        if factor == 0.0:
            raise ZeroDivisionError("Zero division is not permitted.")
        xyz = self._xyz / factor
        return CartesianVector(*xyz)

    def __iadd__(self, other: 'CartesianVector') -> 'CartesianVector':
        """In-place vector addition."""
        if not isinstance(other, CartesianVector):
            raise TypeError(f"Only addition by {type(self)} is permitted.")
        self._xyz += other._xyz
        return self

    def __isub__(self, other: 'CartesianVector') -> 'CartesianVector':
        """In-place vector subtraction."""
        if not isinstance(other, CartesianVector):
            raise TypeError(f"Only subtraction by {type(self)} is permitted.")
        self._xyz -= other._xyz
        return self

    def __imul__(self, factor: Number) -> 'CartesianVector':
        """In-place scalar multiplication."""
        if not isinstance(factor, (int, float)):
            raise TypeError("Only scalar multiplication is permitted.")
        self._xyz *= factor
        return self

    def __itruediv__(self, factor: Number) -> 'CartesianVector':
        """In-place scalar division."""
        if not isinstance(factor, (int, float)):
            raise TypeError("Only scalar division is permitted.")
        if factor == 0.0:
            raise ZeroDivisionError("Zero division is not permitted.")
        self._xyz /= factor
        return self

    def __rmul__(self, factor: Number) -> 'CartesianVector':
        """Right scalar multiplication."""
        if not isinstance(factor, (int, float)):
            raise TypeError("Only scalar multiplication is permitted.")
        xyz = self._xyz * factor
        return CartesianVector(*xyz)

    def dot(self, other: 'CartesianVector') -> Number:
        """Return the dot product of two vectors."""
        if not isinstance(other, CartesianVector):
            raise TypeError("Dot products are only valid with other vectors.")
        return self._xyz.dot(other._xyz)

    def norm(self) -> float:
        """Return the norm of a vector."""
        return np.sqrt(self._xyz.dot(self._xyz))

    def norm_squared(self) -> Number:
        """Return the norm of a vector squared."""
        return self._xyz.dot(self._xyz)

    def cross(self, other: 'CartesianVector') -> 'CartesianVector':
        """Return the cross product of two vectors."""
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return CartesianVector(x, y, z)

    @property
    def x(self) -> Number:
        """Return the first element of the vector."""
        return self._xyz[0]

    @property
    def y(self) -> Number:
        """Return the second element of the vector, if present."""
        return self._xyz[1]

    @property
    def z(self) -> Union[Number, None]:
        """Return the third element of the vector, if present."""
        return self._xyz[2]

    @x.setter
    def x(self, value: Number) -> None:
        """Set the first element of the vector."""
        if not isinstance(value, (int, float)):
            raise TypeError("Invalid x-coordinate type.")
        self._xyz[0] = value

    @y.setter
    def y(self, value: Number) -> None:
        """Set the second element of the vector."""
        if not isinstance(value, (int, float)):
            raise TypeError("Invalid y-coordinate type.")
        self._xyz[1] = value

    @z.setter
    def z(self, value: Number) -> None:
        """Set the third element of the vector."""
        if not isinstance(value, (int, float)):
            raise TypeError("Invalid z-coordinate type.")
        self._xyz[2] = value

    @property
    def dtype(self) -> T:
        """Return the underlying data-type of the vector."""
        return int if self._xyz.dtype == np.int64 else float

    def astype(self, dtype: T) -> 'CartesianVector':
        """Change the underlying data-type."""
        self._xyz = self._xyz.astype(dtyp=dtype)
        return self

    def is_integer(self) -> bool:
        """Return whether all elements are integers."""
        return all(isinstance(v, int) or v.is_integer() for v in self._xyz)

