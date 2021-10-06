import os

import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, Matrix

from pyPDEs.material import CrossSections
from modules.neutron_diffusion.analytic import AnalyticSolution, load

xs = CrossSections()
xs.read_from_xs_file("xs/three_grp_us.cxs", density=0.05)

rf = 6.0
r = symbols("r")
ics = Matrix([1.0 - r**2/rf**2, 1.0 - r**2/rf**2, 0.0])

dr = rf / 100.0
r = np.linspace(0.5*dr, rf - 0.5*dr, 1001)

exact = AnalyticSolution(xs, ics, rf, "SPHERICAL", max_n_modes=100)
exact.execute()
exact.plot_eigenfunction(0, 1, r)


path = os.path.dirname(os.path.abspath(__file__))
fpath = os.path.join(path, "test.obj")
exact.save(fpath)

if not os.path.isfile(fpath):
    raise FileNotFoundError(
        f"Save method failed. {fpath} not found.")

exact: AnalyticSolution = load(fpath)
exact.plot_eigenfunction(0, 1, r)

os.system(f"rm {fpath}")

plt.show()
