from numpy import ndarray
from typing import List



def sigma_a_material_1(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.095 - 1.1)
    else:
        return 1.095


def sigma_a_material_2(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.09 - 1.1)
    elif 0.6 < t <= 1.0:
        return 1.09
    elif 1.0 < t <= 1.7:
        f = (t - 1.0) / 0.7
        return 1.09 + f*(1.1 - 1.09)
    else:
        return 1.1


def sigma_a_material_3(g: int, x: List[float], sigma_a: float) -> float:
    t = x[0]
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.105 - 1.1)
    else:
        return 1.105


xs_vals = {"n_groups": 1, "n_precursors": 1,
           "D": [1.0], "sigma_t": [1.1], "sigma_f": [1.1],
           "transfer_matrix": [[0.0]],
           "velocity": [1000.0],
           "nu_prompt": [0.994], "nu_delayed": [0.006],
           "precursor_lambda": [0.1]}

tolerance = 1.0e-12
max_iterations = int(5.0e4)

__all__ = ["sigma_a_material_1",  "sigma_a_material_2",
           "sigma_a_material_3", "xs_vals", "tolerance",
           "max_iterations"]
