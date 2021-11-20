import numpy as np
from math import factorial, sqrt


def legendre(n: int, x: float) -> float:
    """
    Evaluate Legendre polynomials.

    Parameters
    ----------
    n : int
        The Legendre polynomial order.
    x : float
        The point to evaluate the Legendre polynomial.

    Returns
    -------
    float
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        n = n - 1
        Pn = legendre(n, x)
        Pnm1 = legendre(n - 1, x)
        Pnp1 = (2.0*n + 1.0)*x * Pn - n * Pnm1
        return Pnp1 / (n + 1.0)


def associated_legendre(ell: int, m: int, x: float) -> float:
    """
    Evaluate associated Legendre polynomials.

    Parameters
    ----------
    ell : int
    m : int
    x : float

    Returns
    -------
    float
    """
    if abs(m) > ell:
        return 0.0
    if ell == 0:
        return 1.0
    elif ell == 1 and m == 0:
        return x
    elif ell == m:
        ell = ell - 1
        c = -(2.0*ell + 1.0) * np.sqrt(1.0 - x**2)
        return c * associated_legendre(ell, ell, x)
    else:
        ell = ell - 1
        Plm = associated_legendre(ell, m, x)
        Plm1m = associated_legendre(ell-1, m, x)
        Plp1m = (2.0*ell + 1.0)*x * Plm - (ell + m) * Plm1m
        return Plp1m / (ell - m + 1.0)


def spherical_harmonics(ell: int, m: int,
                        varphi: float, theta: float) -> float:
    """
    Evaluate spherical harmonic functions.

    Parameters
    ----------
    ell : int
    m : int
    varphi : float
    theta : float

    Returns
    -------
    float
    """
    Plm = associated_legendre(ell, m, np.cos(theta))

    if m == 0:
        return Plm
    else:
        fm = factorial(ell - abs(m))
        fp = factorial(ell + abs(m))
        c = (-1.0)**(abs(m)) * sqrt(2.0 * fm/fp) * Plm
        if m < 0:
            return c * np.sin(abs(m) * varphi)
        else:
            return c * np.cos(abs(m) * varphi)
