import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from studies.utils import *
from pyROMs.pod import POD

warnings.filterwarnings('ignore')

########################################
# Parse the data
########################################
study = int(sys.argv[1])
dataset = get_data('twigl', study)
n_params = dataset.n_parameters

var = 'power_density'
interior_only = False
test_size = 0.2 if n_params == 1 or not interior_only else 0.5
eps = 1.0 if n_params == 1 else 10.0

splits = dataset.train_test_split(variables=var,
                                  test_size=test_size, seed=12,
                                  interior_only=interior_only)
X_train, X_test, Y_train, Y_test = splits

rom_info = {'interpolant': [], 'n_modes': [], 'mean_error': [],
            'max_error': [], 'min_error': []}

interps = ['rbf_linear', 'rbf_thin_plate_spline',
           'rbf_cubic', 'rbf_quintic', 'rbf_gaussian',
           'linear', 'nearest']
for interp in interps:

    # Construct POD model, predict test data
    start_time = time.time()
    svd_rank = 0
    pod = POD(svd_rank=svd_rank)
    pod.fit(X_train, Y_train, interp, epsilon=eps)
    contruction_time = time.time() - start_time

    # Predict results
    X_pred = []
    avg_predict_time = 0.0
    for y_test in Y_test:
        start_time = time.time()
        X_pred.append(pod.predict(y_test))
        avg_predict_time += time.time() - start_time
    avg_predict_time /= len(Y_test)

    # Format datasets
    X_pred = dataset.unstack_simulation_vector(X_pred)
    X_test = dataset.unstack_simulation_vector(X_test)

    # Compute simulation errors
    errors = np.zeros(len(X_test))
    for i in range(len(X_test)):
        errors[i] = norm(X_test[i]-X_pred[i])/norm(X_test[i])

    rom_info['interpolant'].append(interp)
    rom_info['n_modes'].append(pod.n_modes)
    rom_info['mean_error'].append(np.mean(errors))
    rom_info['max_error'].append(np.max(errors))
    rom_info['min_error'].append(np.min(errors))

msg = "\\begin{tabular}{|c|c|c|c|}" \
      "\n\t\hline" \
      "\n\t\\textbf{Interpolation Method} & \\textbf{Mean Error} & " \
      "\\textbf{Max Error} & \\textbf{Min Error} \\\\ \hline"
for i in range(len(interps)):
    interp = interps[i].split('_')
    if interp[0] == 'rbf':
        interp[0] = interp[0].upper()
        for w in range(len(interp[1:])):
            interp[w+1] = interp[w+1].capitalize()
        interp = " ".join(interp[1:]) + " RBF"
        if "Gaussian" in interp:
            interp += f", $\\epsilon$ = {eps:.2e}"
    else:
        interp = interp[0].capitalize()
        if interp == "Nearest":
            interp += " Neighbor"
        elif interp == "Cubic":
            interp += " Spline"

    msg += f"\n\t\hline {interp} & " \
           f"{rom_info['mean_error'][i]:.3e} & " \
           f"{rom_info['max_error'][i]:.3e} & " \
           f"{rom_info['min_error'][i]:.3e} \\\\"
msg += "\n\t\hline\n\\end{tabular}"
print()
print(msg)
print()
print(f"Number of Snapshots:\t{pod.n_snapshots}")
print(f"Number of Validations:\t{len(X_test)}")
print(f"Number of Modes:\t{pod.n_modes}")
