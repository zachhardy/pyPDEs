"""
Cross section functions for `prince_prototype_1d_fv.py` and
`prince_minicore_1d_fv.py` test problems.
"""
__all__ = ["sigma_t_1", "sigma_t_2", "sigma_t_3"]

def sigma_t_1(t: float, sigt_i: float) -> float:
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.095 - 1.1)
    else:
        return 1.095

def sigma_t_2(t: float, sigt_i: float) -> float:
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

def sigma_t_3(t: float, sigt_i: float) -> float:
    if t <= 0.1:
        return 1.1
    elif 0.1 < t <= 0.6:
        f = (t - 0.1) / 0.5
        return 1.1 + f*(1.105 - 1.1)
    else:
        return 1.105