import matplotlib.pyplot as plt
import numpy as np
from pyPDEs.material import CrossSections

nu = 2.5
xs_fuel = {'n_groups': 2,
           'sigma_t': [10.0, 20.0],
           'transfer_matrix': [[3.0, 1.0], [0.0, 8.0]],
           'chi': [1.0, 0.0],
           'sigma_f': [2.0, 8.0],
           'nu': [nu, nu],
           'velocity': [1000.0, 100.0]}

xs_refl = {'n_groups': 2,
           'sigma_t': [20.0, 15.0],
           'transfer_matrix': [[5.0, 10.0], [0.0, 12.0]],
           'velocity': [1000.0, 100.0]}

fuel = CrossSections()
fuel.read_from_xs_dict(xs_fuel)

refl = CrossSections()
refl.read_from_xs_dict(xs_refl)

r_f = 10.0
r_b = 12.5
d_r = r_b - r_f

def Xp_X(alpha):
    m = np.sqrt(mu2(alpha))
    return m*cot(m*r_f) - 1.0/r_f


def Yp_Y(alpha):
    l = np.sqrt(lambda2(alpha))
    return l*coth(l*r_f) - 1.0/r_f


def Z1p_Z1(alpha):
    B = np.sqrt(Beff2_g0(alpha))
    return -1.0/r_f - B*coth(B*d_r)


def Z2p_Z2(alpha):
    B = np.sqrt(Beff2_g1(alpha))
    return -1.0/r_f - B*coth(B*d_r)


def mu2(alpha):
    b = beta(alpha)
    g = gamma(alpha)
    return 0.5*(-b + np.sqrt(b**2 + 4*g))


def lambda2(alpha):
    b = beta(alpha)
    g = gamma(alpha)
    return 0.5*(b + np.sqrt(b**2 + 4*g))


def beta(alpha):
    sig_a = fuel.sigma_a[1]
    D0, D1 = fuel.D[0], fuel.D[1]
    return RF(alpha)/D0 + sig_a/D1


def gamma(alpha):
    sig_0to1 = fuel.transfer_matrix[0][0][1]
    nusig_f1 = fuel.nu_sigma_f[1]
    sig_a1 = fuel.sigma_a[1]
    D0, D1 = fuel.D[0], fuel.D[1]
    return (sig_0to1*nusig_f1 - sig_a1*RF(alpha)) / (D0*D1)


def x_coupling(alpha):
    sig_0to1 = fuel.transfer_matrix[0][0][1]
    D = fuel.D[1]
    sig_a = fuel.sigma_a[1]
    return sig_0to1 / (D*mu2(alpha) + sig_a)


def y_coupling(alpha):
    sig_0to1 = fuel.transfer_matrix[0][0][1]
    D = fuel.D[1]
    sig_a = fuel.sigma_a[1]
    return sig_0to1 / (sig_a - D*lambda2(alpha))


def z_coupling(alpha):
    sig_0to1 = refl.transfer_matrix[0][0][1]
    D = refl.D[1]
    return sig_0to1 / (D * (Beff2_g1(alpha) - Beff2_g1(alpha)))


def RF(alpha):
    sig_r = fuel.sigma_r[0]
    v = fuel.velocity[0]
    nusig_f = fuel.nu_sigma_f[0]
    return sig_r + alpha/v - nusig_f


def Beff2_g0(alpha):
    sig_r = refl.sigma_r[0]
    v = refl.velocity[0]
    D = refl.D[0]
    return (sig_r + alpha/v) / D


def Beff2_g1(alpha):
    sig_a = refl.sigma_a[1]
    v = refl.velocity[1]
    D = refl.D[1]
    return (sig_a + alpha/v) / D


def cot(x):
    return np.cos(x) / np.sin(x)


def coth(x):
    return np.cosh(x) / np.sinh(x)


def RHS(alpha):
    D_C, D_R = fuel.D, refl.D
    x = x_coupling(alpha)
    y = y_coupling(alpha)
    z = z_coupling(alpha)

    Y_ratio = Yp_Y(alpha)
    Z1_ratio = Z1p_Z1(alpha)
    Z2_ratio = Z2p_Z2(alpha)

    # Numerator
    # Y, Z1 term
    Y_Z1 = D_R[0]*D_C[1]*y - D_C[0]*D_R[1]*z
    Y_Z1 *= Y_ratio * Z1_ratio

    # Y, Z2 term
    Y_Z2 = D_C[0]*D_R[1]*(z - x)
    Y_Z2 *= Y_ratio * Z2_ratio

    # Z1, Z2 term
    Z1_Z2 = D_R[0]*D_R[1]*(x - y)
    Z1_Z2 *= Z1_ratio * Z2_ratio

    # Denominator
    Y = D_C[0]*D_C[1]*(y - x) * Y_ratio
    Z1 = (D_R[0]*D_C[1]*x - D_C[0]*D_R[1]*z) * Z1_ratio
    Z2 = D_C[0]*D_R[1]*(z - y) * Z2_ratio

    return (Y_Z1 + Y_Z2 + Z1_Z2) / (Y + Z1 + Z2)



print(refl.sigma_r[0] * refl.velocity[0])

