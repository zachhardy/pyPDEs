import sys
import time
import warnings

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from studies.utils import *
from pyROMs.pod import POD
from pyROMs.dmd import DMD

warnings.filterwarnings('ignore')

########################################
# Parse the data
########################################
case, study = int(sys.argv[1]), int(sys.argv[2])
dataset = get_data('three_group_sphere', case, study)

splits = dataset.train_test_split(test_size=0.2, seed=12,
                                  interior_only=False)
X_train, X_test, Y_train, Y_test = splits

# Construct POD model, predict test data
start_time = time.time()
tau = 1.0e-8
pod = POD(svd_rank=1.0-tau)
pod.fit(X_train, Y_train, 'rbf', epsilon=5.0)
contruction_time = time.time() - start_time
pod.print_summary()

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

# Print aggregated POD results
msg = f"===== Summary of {len(errors)} POD Queries ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Number of Training Samples:\t\t{len(X_train)}")
print(f"Number of POD Modes:\t\t\t{pod.n_modes}")
print(f"Average POD Reconstruction Error:\t{np.mean(errors):.3e}")
print(f"Maximum POD Reconstruction Error:\t{np.max(errors):.3e}")
print(f"Minimum POD Reconstruction Error:\t{np.min(errors):.3e}")
print(f"Construction Time:\t\t\t{contruction_time:.3e} s")
print(f"Average Query Time:\t\t\t{avg_predict_time:.3e} s")

avg_dmd_time = 0.0
errors = np.zeros(len(X_pred))
for i in range(len(X_pred)):
    t_start = time.time()
    dmd = DMD(svd_rank=1.0-tau*1.0e-2).fit(X_pred[i])
    avg_dmd_time += (time.time()-t_start)/len(X_pred)

    x_dmd = dmd.reconstructed_data
    errors[i] = norm(X_test[i] - x_dmd)/norm(X_test[i])

# Print aggregated DMD results
msg = f"===== Summary of {errors.size} DMD Models ====="
header = "=" * len(msg)
print("\n".join(["", header, msg, header]))
print(f"Number of Training Samples:\t\t{dmd.n_snapshots}")
print(f"Number of DMD Modes:\t\t\t{dmd.n_modes}")
print(f"Average DMD Reconstruction Error:\t{np.mean(errors):.3e}")
print(f"Maximum DMD Reconstruction Error:\t{np.max(errors):.3e}")
print(f"Minimum DMD Reconstruction Error:\t{np.min(errors):.3e}")
print(f"Average DMD Execution Time:\t\t{avg_dmd_time:.3e} s")
print(f"Average POD-DMD Query Time:\t\t{avg_predict_time + avg_dmd_time:.3e} s")
