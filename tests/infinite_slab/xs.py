from numpy import ndarray
from typing import List

__all__ = ['sigma_a_ramp_up', 'sigma_a_ramp_down', 'sigma_a_fast_ramp_down',
           'xs_material_0_and_2', 'xs_material_1', 'tolerance',
           'max_iterations']


def sigma_a_ramp_up(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1 and 0.0 <= t <= 1.0:
        return sigma_a * (1.0 + t * 0.03)
    elif g == 1 and t > 1.0:
        return 1.03 * sigma_a
    else:
        return sigma_a


def sigma_a_ramp_down(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1 and 0.0 < t <= 1.0:
        return sigma_a * (1.0 - t * 0.01)
    elif g == 1 and t > 1.0:
        return 0.99 * sigma_a
    else:
        return sigma_a


def sigma_a_fast_ramp_down(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1 and 0.0 <= t <= 0.01:
        return sigma_a * (1.0 - t/0.01 * 0.05)
    elif g == 1 and t > 0.01:
        return 0.95 * sigma_a
    else:
        return sigma_a


decay = [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]

beta_i = [02.5e-4, 1.64e-3, 1.47e-3, 2.96e-3, 8.6e-4, 3.2e-4]
beta_tot = sum(beta_i)

nu = 1.0
nu_prompt = (1.0 - beta_tot) * nu
nu_delayed = beta_tot * nu

gamma_i = [b / nu_delayed for b in beta_i]

nu_prompt = [nu_prompt] * 2
nu_delayed = [nu_delayed] * 2

chi_prompt = [1.0, 0.0]
chi_delayed = [[1.0] * 6, [0.0] * 6]

velocity = [1.0e7, 3.0e5]

xs_material_0_and_2 = \
    {'n_groups': 2, 'n_precursors': 6,
     'D': [1.5, 0.5], 'sigma_t': [0.026, 0.18],
     'sigma_f': [0.01, 0.2],
     'transfer_matrix': [[0.0, 0.015], [0.0, 0.0]],
     'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
     'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
     'precursor_lambda': decay, 'precursor_yield': gamma_i,
     'velocity': velocity}

xs_material_1 = \
    {'n_groups': 2, 'n_precursors': 6,
     'D': [1.0, 0.5], 'sigma_t': [0.02, 0.08],
     'sigma_f': [0.005, 0.099],
     'transfer_matrix': [[0.0, 0.01], [0.0, 0.0]],
     'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
     'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
     'precursor_lambda': decay, 'precursor_yield': gamma_i,
     'velocity': velocity}

tolerance = 1.0e-12
max_iterations = int(1.0e4)
