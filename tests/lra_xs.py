"""
Cross section functions for `lra.py` test problem.
"""
__all__ = ["fuel_1_with_rod", "fuel_1_without_rod",
           "fuel_2_with_rod", "fuel_2_without_rod",
           "reflector", "sigma_t_generic"]

from numpy import ndarray

def sigma_t_generic(g: int, t: float, sigma_t: ndarray) -> float:
   if g == 1:
      if t <= 2.0:
         return sigma_t[g] * (1.0 - 0.0606184*t)
      else:
         return sigma_t[g] * 0.8787631
   else:
      return sigma_t[g]

beta_i = [0.0054, 0.001087]
beta = sum(beta_i)

gamma = [b / beta for b in beta_i]

decay = [0.0654, 1.35]

nu = 2.43
nu_prompt = [(1.0 - beta) * nu] * 2
nu_delayed = [beta * nu] * 2

chi_prompt = [1.0, 0.0]
chi_delayed = [[1.0, 1.0], [0.0, 0.0]]

velocity = [3.0e7, 3.0e5]

fuel_1_with_rod = \
    {"n_groups": 2, "n_precursors": 2,
     "D": [1.255, 0.211], "sigma_a": [0.008252, 0.1003],
     "sigma_f": [0.004602/nu, 0.1091/nu],
     "transfer_matrix":[[0.0, 0.02533], [0.0, 0.0]],
     "nu_prompt": nu_prompt, "nu_delayed": nu_delayed,
     "chi_prompt": chi_prompt, "chi_delayed": chi_delayed,
     "precursor_lambda": decay, "precursor_yield": gamma,
     "velocity": velocity}

fuel_1_without_rod = \
    {"n_groups": 2, "n_precursors": 2,
     "D": [1.268, 0.1902], "sigma_a": [0.007181, 0.07047],
     "sigma_f": [0.004609/nu, 0.08675/nu],
     "transfer_matrix":[[0.0, 0.02767], [0.0, 0.0]],
     "nu_prompt": nu_prompt, "nu_delayed": nu_delayed,
     "chi_prompt": chi_prompt, "chi_delayed": chi_delayed,
     "precursor_lambda": decay, "precursor_yield": gamma,
     "velocity": velocity}

fuel_2_with_rod = \
    {"n_groups": 2, "n_precursors": 2,
     "D": [1.259, 0.2091], "sigma_a": [0.008002, 0.08344],
     "sigma_f": [0.004663/nu, 0.1021/nu],
     "transfer_matrix":[[0.0, 0.02617], [0.0, 0.0]],
     "nu_prompt": nu_prompt, "nu_delayed": nu_delayed,
     "chi_prompt": chi_prompt, "chi_delayed": chi_delayed,
     "precursor_lambda": decay, "precursor_yield": gamma,
     "velocity": velocity}

fuel_2_without_rod = \
    {"n_groups": 2, "n_precursors": 2,
     "D": [1.259, 0.2091], "sigma_a": [0.008002, 0.073324],
     "sigma_f": [0.004663/nu, 0.1021/nu],
     "transfer_matrix":[[0.0, 0.02617], [0.0, 0.0]],
     "nu_prompt": nu_prompt, "nu_delayed": nu_delayed,
     "chi_prompt": chi_prompt, "chi_delayed": chi_delayed,
     "precursor_lambda": decay, "precursor_yield": gamma,
     "velocity": velocity}

reflector = \
    {"n_groups": 2, "n_precursors": 0,
     "D": [1.257, 0.1592], "sigma_a": [0.0006034, 0.01911],
     "transfer_matrix":[[0.0, 0.04754], [0.0, 0.0]],
     "velocity": velocity}

