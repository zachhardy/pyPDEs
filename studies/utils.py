"""
Support functions for studies.
"""
__all__ = ['setup_directory', 'setup_range']

import os
import numpy as np


def setup_directory(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    elif len(os.listdir(path)) > 0:
        os.system(f'rm -r {path}/*')


def setup_range(ref: float, var: float, N: int):
    return ref * (1.0 + var*np.linspace(-1.0, 1.0, N))
