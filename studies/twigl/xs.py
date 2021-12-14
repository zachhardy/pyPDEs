from numpy import ndarray
from typing import List

__all__ = ['xs_material_0', 'xs_material_1',
           'tolerance', 'max_iterations']


trnsfr = [[0.0, 0.01], [0.0, 0.0]]

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
        'transfer_matrix': trnsfr,
        'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
        'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
        'precursor_lambda': decay, 'velocity': velocity}

xs_material_1 = \
       {'n_groups': 2, 'n_precursors': 1,
        'D': [1.3, 0.5], 'sigma_a': [0.008, 0.05],
        'sigma_f': [0.003/nu, 0.06/nu],
        'transfer_matrix': trnsfr,
        'nu_prompt': nu_prompt, 'nu_delayed': nu_delayed,
        'chi_prompt': chi_prompt, 'chi_delayed': chi_delayed,
        'precursor_lambda': decay, 'velocity': velocity}

tolerance = 1.0e-12
max_iterations = int(1.0e4)
