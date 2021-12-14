import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from readers import NeutronicsSimulationReader


script_path = os.path.dirname(os.path.abspath(__file__))

# Get inputs
case = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if case == 0:
    study_name = 'multiplier'
elif case == 1:
    study_name = 'duration'
elif case == 2:
    study_name = 'interface'
elif case == 3:
    study_name = 'multiplier_duration'
elif case == 4:
    study_name = 'multiplier_interface'
elif case == 5:
    study_name = 'duration_interface'
elif case == 6:
    study_name = 'multiplier_duration_interface'


base = os.path.join(script_path, f'outputs/subcritical/{study_name}')

try:
    if len(sys.argv) != 3:
        raise AssertionError(
            'There must be a command line argument to point to '
            'the study and simulation number.\n')

    if sys.argv[2] == 'reference':
        path = os.path.join(base, sys.argv[1])
    else:
        if int(sys.argv[2]) > len(os.listdir(base)):
            raise ValueError(
                'The provided index must match a test case.')

        path = os.path.join(base, sys.argv[2].zfill(3))
        if not os.path.isdir(path):
            raise NotADirectoryError('Invalid path to test case.')

except BaseException as err:
    print(); print(err.args[0]); print()
    sys.exit()

params_path = os.path.join(base, 'params.txt')
params = np.loadtxt(params_path)
if params.ndim == 1:
    params = params.reshape(-1, 1)

param_names = base.split('/')[-1].split('_')

sim = NeutronicsSimulationReader(path)
sim.read_simulation_data()

sim.plot_flux_moments(0, 1, [0.0, 1.0, 2.0], grouping='time')

title = ''
for n, name in enumerate(param_names):
    suffix = ''
    if name == 'multiplier':
        name = '$\sigma_a$ Increase'
        suffix = '%'
    if name == 'duration':
        name = 'Ramp Duration'
        suffix = 'sec'
    if name == 'interface':
        name = 'Interface Location'
        suffix = 'cm'

    tmp = f'{name} = {params[int(sys.argv[2])][n]:.3f} {suffix}'
    if n == 0:
        title += tmp
    else:
        title += '\n'+tmp

plt.title(title)
plt.tight_layout()
plt.show()

# from rom.dmd import DMD
# X = sim.create_simulation_matrix()
# dmd = DMD(svd_rank=1.0-1.0e-12)
# dmd.fit(X, sim.times)
#
# dmd.plot_error_decay()
# dmd.plot_timestep_errors()
# plt.show()

