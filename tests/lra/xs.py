import numpy as np
from numpy import ndarray
from typing import List

__all__ = ['fuel_1_with_rod', 'fuel_1_without_rod',
           'fuel_2_with_rod', 'fuel_2_without_rod', 'fuel_2_control',
           'reflector', 'sigma_a_with_rod', 'sigma_a_without_rod']

delta = 0.8787631 - 1.0
t_ramp = 2.0
gamma = 3.034e-3


def sigma_a_with_rod(g: int, x: List[float], sigma_a: float) -> float:
    assert len(x) == 4, 'There must be 4 variables in `x` input.'
    t, T, T0 = x[0], x[1], x[2]

    if g == 0:
        return sigma_a * (1.0 + gamma*(np.sqrt(T) - np.sqrt(T0)))
    elif g == 1:
        if t <= t_ramp:
            return sigma_a * (1.0 + t/t_ramp*delta)
        else:
            return (1.0 + delta)*sigma_a
    else:
        return sigma_a


def sigma_a_without_rod(g: int, x: List[float], sigma_a: float) -> float:
    assert len(x) == 4, 'There must be 4 variables in `x` input.'
    t, T, T0 = x[0], x[1], x[2]

    if g == 0:
        return sigma_a * (1.0 + gamma*(np.sqrt(T) - np.sqrt(T0)))
    else:
        return sigma_a


beta_i = [0.0054, 0.001087]
beta = sum(beta_i)

yield_i = [b / beta for b in beta_i]

decay = [0.0654, 1.35]

nu = 2.43
nu_prompt = [(1.0 - beta) * nu] * 2
nu_delayed = [beta * nu] * 2

chi_prompt = [1.0, 0.0]
chi_delayed = [[1.0, 1.0], [0.0, 0.0]]

velocity = [3.0e7, 3.0e5]

buckling = 1.0e-4

fuel_1_with_rod = \
    {'n_groups': 2, 'n_precursors': 2,
     'D': [1.255, 0.211], 'sigma_a': [0.008252, 0.1003],
     'buckling': buckling, 'sigma_f': [0.004602/nu, 0.1091/nu],
     'transfer_matrix': [[0.0, 0.02533], [0.0, 0.0]],
     'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
     'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
     'precursor_lambda': decay, 'precursor_yield': yield_i,
     'velocity': velocity}

fuel_1_without_rod = \
    {'n_groups': 2, 'n_precursors': 2,
     'D': [1.268, 0.1902], 'sigma_a': [0.007181, 0.07047],
     'buckling': buckling, 'sigma_f': [0.004609/nu, 0.08675/nu],
     'transfer_matrix': [[0.0, 0.02767], [0.0, 0.0]],
     'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
     'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
     'precursor_lambda': decay, 'precursor_yield': yield_i,
     'velocity': velocity}

fuel_2_with_rod = \
    {'n_groups': 2, 'n_precursors': 2,
     'D': [1.259, 0.2091], 'sigma_a': [0.008002, 0.08344],
     'buckling': buckling, 'sigma_f': [0.004663/nu, 0.1021/nu],
     'transfer_matrix': [[0.0, 0.02617], [0.0, 0.0]],
     'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
     'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
     'precursor_lambda': decay, 'precursor_yield': yield_i,
     'velocity': velocity}

from copy import deepcopy
fuel_2_control = deepcopy(fuel_2_with_rod)
fuel_2_control['sigma_a'][1] *= 0.8787631


fuel_2_without_rod = \
    {'n_groups': 2, 'n_precursors': 2,
     'D': [1.259, 0.2091], 'sigma_a': [0.008002, 0.073324],
     'buckling': buckling, 'sigma_f': [0.004663/nu, 0.1021/nu],
     'transfer_matrix': [[0.0, 0.02617], [0.0, 0.0]],
     'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
     'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
     'precursor_lambda': decay, 'precursor_yield': yield_i,
     'velocity': velocity}

reflector = \
    {'n_groups': 2, 'n_precursors': 0,
     'D': [1.257, 0.1592], 'sigma_a': [0.0006034, 0.01911],
     'buckling': buckling,
     'transfer_matrix':[[0.0, 0.04754], [0.0, 0.0]],
     'velocity': velocity}
