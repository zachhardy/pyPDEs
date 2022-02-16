from numpy import ndarray
from typing import List

__all__ = ['xs_material_0', 'xs_material_1',
           'sigma_a_ramp', 'sigma_a_step', 'tolerance',
           'max_iterations']


def sigma_a_ramp(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1:
        if 0.0 <= t <= 0.2:
            return sigma_a*(1.0 + t/0.2*(0.97667 - 1.0))

        else:
            return 0.97667*sigma_a
    else:
        return sigma_a


def sigma_a_step(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if g == 1 and t > 0.0:
        return 0.97667*sigma_a
    else:
        return sigma_a


nu = 2.43
beta = 0.0075
nu_prompt = [(1.0 - beta) * nu] * 2
nu_delayed = [beta * nu] * 2

chi_prompt = [1.0, 0.0]
chi_delayed = [[1.0], [0.0]]

decay = [0.08]

velocity = [1.0e7, 2.0e5]

xs_material_0 = \
       {'n_groups': 2, 'n_precursors': 1,
        'D': [1.4, 0.4], 'sigma_a': [0.01, 0.15],
        'sigma_f': [0.007/nu, 0.2/nu],
        'transfer_matrix': [[0.0, 0.01], [0.0, 0.0]],
        'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
        'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
        'precursor_lambda': decay, 'velocity': velocity}

xs_material_1 = \
       {'n_groups': 2, 'n_precursors': 1,
        'D': [1.3, 0.5], 'sigma_a': [0.008, 0.05],
        'sigma_f': [0.003/nu, 0.06/nu],
        'transfer_matrix': [[0.0, 0.01], [0.0, 0.0]],
        'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
        'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
        'precursor_lambda': decay, 'velocity': velocity}

tolerance = 1.0e-12
max_iterations = int(1.0e4)
