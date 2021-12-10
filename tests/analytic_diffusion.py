import os

import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, Matrix

from pyPDEs.material import CrossSections
from modules.neutron_diffusion.analytic import AnalyticSolution, load

xs = CrossSections()
xs.read_from_xs_file('xs/three_grp_us.cxs', density=0.05)

r_b = 6.0
r = symbols('r')
ics = Matrix([1.0 - r ** 2 / r_b ** 2,
              1.0 - r ** 2 / r_b ** 2,
              0.0])

dr = r_b / 100.0
r = np.linspace(0.5 * dr, r_b - 0.5 * dr, 1001)

exact = AnalyticSolution(xs, ics, r_b, 'spherical', max_n_modes=1000)
exact.execute()

exact.get_mode(0, method='amp').plot_eigenfunction(r)
exact.get_mode(1, method='amp').plot_eigenfunction(r)

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, 'three_group_sphere', 'transient')
exact.save(path + f'/sphere{int(r_b)}cm.obj')

plt.show()
