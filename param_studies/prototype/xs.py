from numpy import ndarray
from typing import List


xs_vals = {"n_groups": 1, "n_precursors": 1,
           "D": [1.0], "sigma_t": [1.1], "sigma_f": [1.1],
           "transfer_matrix": [[0.0]],
           "velocity": [1000.0],
           "nu_prompt": [0.994], "nu_delayed": [0.006],
           "precursor_lambda": [0.1]}

tolerance = 1.0e-12
max_iterations = int(5.0e4)

__all__ = ["xs_vals", "tolerance", "max_iterations"]
